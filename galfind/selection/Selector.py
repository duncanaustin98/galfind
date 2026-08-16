"""Galaxy sample selection criteria and selectors.

This module defines the abstract and concrete selector classes used to
apply well-defined selection criteria to galaxies in a catalogue. Selectors
encapsulate individual selection criteria (e.g., redshift cuts, signal-to-noise
cuts, masking criteria) that can be applied to individual galaxies or entire
catalogues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
import json
from astropy.coordinates import SkyCoord
import h5py
from astropy.io import fits
from pathlib import Path
from astropy.utils.masked import Masked
from photutils.centroids import centroid_2dg
from photutils.aperture import CircularAperture
from scipy import ndimage
from tqdm import tqdm
import logging

from typing import TYPE_CHECKING, Any, List, Union, Callable, Optional, Dict
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

from ..utils import useful_funcs_austind as funcs
from .. import galfind_logger, config, wav_lyman_lim
from . import Galaxy, Catalogue, Catalogue_Base, Instrument, SED_code, Depths, Multiple_Filter
from .Instrument import expected_instr_bands
from .Morphology import fwhm_nircam

class Selector(ABC):
    """Abstract base class defining the galaxy-sample-selection interface.

    A `Selector` encapsulates a single, well-defined selection criterion
    (e.g. a redshift cut, a signal-to-noise cut, a masking criterion) that
    can be applied either to an individual `Galaxy` or to an entire
    `Catalogue`. Calling an instance (`__call__`) evaluates the criterion
    for every galaxy in the target object, annotating each `Galaxy` with a
    boolean selection flag (keyed by `name`) and any diagnostic selection
    keyword values in `gal.selection_flags`/`gal.selection_kwargs`, and
    (when applied to a `Catalogue`) appends the corresponding columns to
    the output table.

    Subclasses must implement the following abstract members to fully
    specify the selection contract:

    - `name` (property): unique, human-readable name identifying this
      selection (used as the column/flag name).
    - `_selection_name` (property): short internal name for the criterion.
    - `_include_kwargs` (property): names of keyword arguments required
      to be present in `kwargs` for this selector.
    - `_assertions`: sanity checks run at the end of `__init__` to
      validate the selector's configuration.
    - `_selection_criteria`: evaluates the criterion for a single galaxy,
      returning whether it passes and a `dict` of diagnostic keyword
      values.
    - `_check_phot_exists`, `_check_SED_fit_exists`,
      `_check_morph_fit_exists`: verify that the photometry, SED-fitting,
      and morphology-fitting data required to evaluate the criterion are
      loaded for a given galaxy.

    Subclasses may additionally override `_failure_criteria` (an early
    "automatic fail" check, e.g. for objects with no possibility of
    passing regardless of `_selection_criteria`) which defaults to
    always returning `False`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity` or `None`
        Aperture diameter used to extract the photometry this selector
        acts on, if applicable.
    SED_fitter : `SED_code` or `None`
        SED-fitting code/run whose results this selector acts on, if
        applicable.
    morph_fitter : `Type`[`Morphology_Fitter`] or `None`
        Morphology-fitting code/run whose results this selector acts on,
        if applicable.
    fit_filterset : `Multiple_Filter`, optional
        Filter set over which the relevant fit (SED or morphology) was
        performed. Default is `None`.
    **kwargs
        Additional selector-specific keyword arguments; must include all
        names listed in `_include_kwargs`.

    Attributes
    ----------
    aper_diam : `astropy.units.Quantity` or `None`
        Aperture diameter as given at construction.
    SED_fitter : `SED_code` or `None`
        SED-fitting code/run as given at construction.
    morph_fitter : `Type`[`Morphology_Fitter`] or `None`
        Morphology-fitting code/run as given at construction.
    fit_filterset : `Multiple_Filter` or `None`
        Filter set as given at construction.
    kwargs : `dict`
        Selector-specific keyword arguments as given at construction.

    Raises
    ------
    AssertionError
        If `kwargs` is missing any of the keys required by
        `_include_kwargs`, if `fit_filterset` is not a `Multiple_Filter`
        instance, or if `_assertions` fails.
    """

    def __init__(
        self: Self,
        aper_diam: Optional[u.Quantity],
        SED_fitter: Optional[SED_code],
        morph_fitter: Optional[Type[Morphology_Fitter]],
        fit_filterset: Optional[Multiple_Filter] = None,
        **kwargs,
    ):
        assert (key in kwargs.keys() for key in self._include_kwargs), \
            galfind_logger.critical(
                f"Selection {self.__class__.__name__} given {kwargs=}" + \
                f" missing required keys = {self._include_kwargs}."
            )
        if fit_filterset is not None:
            assert isinstance(fit_filterset, Multiple_Filter), \
                galfind_logger.critical(
                    f"{fit_filterset=} must be a Multiple_Filter object."
                )
        self.aper_diam = aper_diam
        self.SED_fitter = SED_fitter
        self.morph_fitter = morph_fitter
        self.fit_filterset = fit_filterset
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
        """`str`: Unique name of this selection, used as the flag/column name."""
        pass

    @property
    @abstractmethod
    def _selection_name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def _include_kwargs(self: Self) -> List[str]:
        pass

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names and dtypes of diagnostic selection kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            The names of the diagnostic keyword values stored in
            `gal.selection_kwargs[self.name]` and their corresponding
            output dtypes, used when appending catalogue columns. The
            base implementation returns two empty lists (no diagnostic
            kwargs).
        """
        return [], []

    @abstractmethod
    def _assertions(self: Self) -> bool:
        pass

    #@abstractmethod
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        # always pass by default
        return False

    @abstractmethod
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
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
        """Apply this selection criterion to a galaxy or catalogue.

        Dispatches to `_call_gal` for a single `Galaxy` or `_call_cat`
        for a `Catalogue`, annotating the target object's selection
        flags and diagnostic kwargs with the result of this criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue_Base`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on (and return) a deep copy of `object`
            rather than modifying it in place. Default is `True`.
        *args, **kwargs
            Additional arguments forwarded to `_selection_criteria` and
            `_failure_criteria`.

        Returns
        -------
        `Galaxy`, `Catalogue_Base`, or `None`
            The (possibly copied) annotated object, or the catalogue
            cropped to the selection if `return_copy` is `True`. Returns
            `None` if `return_copy` is `False`.

        Raises
        ------
        ValueError
            If `object` is neither a `Galaxy` nor a `Catalogue_Base`
            subclass instance.
        """
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
    ) -> Optional[Galaxy]:
        if return_copy:
            gal_ = deepcopy(gal)
        else:
            gal_ = gal
        selection_name = self.name
        if (
            not selection_name in gal_.selection_flags.keys() or \
            not all(
                kwarg_name in gal_.selection_kwargs[selection_name].keys()
                for kwarg_name, kwarg_dtype in zip(*self.select_kwarg_names_dtypes)
            )
        ):
            if self._failure_criteria(gal_, *args, **kwargs) \
                    or not self._check_phot_exists(gal_) \
                    or not self._check_SED_fit_exists(gal_) \
                    or not self._check_morph_fit_exists(gal_):
                gal_.selection_flags[selection_name] = False
                gal_.selection_kwargs[selection_name] = {
                    kwarg_name: None if kwarg_dtype not in [int] else 0 \
                    for kwarg_name, kwarg_dtype in zip(*self.select_kwarg_names_dtypes)
                }
            else:
                selected, select_kwargs = self._selection_criteria(gal_, *args, **kwargs)
                try:
                    assert all(
                        kwarg_name in select_kwargs.keys()
                        for kwarg_name, kwarg_dtype in zip(*self.select_kwarg_names_dtypes)
                    ), galfind_logger.critical(
                        f"{repr(self)} selection criteria must return select_kwargs " + \
                        f"with keys = {str(self.select_kwarg_names_dtypes[0])}!"
                    )
                except:
                    breakpoint()
                gal_.selection_flags[selection_name] = selected
                gal_.selection_kwargs[selection_name] = select_kwargs
        if return_copy:
            return gal_

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Catalogue]:
        self._assert_cat(cat)
        [
            self._call_gal(gal, return_copy = False, *args, **kwargs) for gal \
            in tqdm(cat, total = len(cat), desc = f"Selecting {self.name}", \
            disable = galfind_logger.getEffectiveLevel() > logging.INFO)
        ]
        cat._append_property_to_tab(
            self.name,
            hdu = "SELECTION",
            dtype = bool,
        )
        for kwarg_name, kwarg_dtype in zip(*self.select_kwarg_names_dtypes):
            save_name = f"{self.name}__{kwarg_name}"
            cat._append_property_to_tab(
                save_name,
                hdu = "SELECTION",
                dtype = kwarg_dtype,
            )
        if return_copy:
            cat_copy = deepcopy(cat)
            return cat_copy.crop(self)
        else:
            return cat
        
    def _assert_cat(self: Self, cat: Catalogue) -> None:
        if self.SED_fitter is not None:
            # ensure results have been loaded for 
            # at least 1 galaxy in the catalogue
            assert any(self._check_SED_fit_exists(gal) for gal in cat), \
                galfind_logger.critical(
                    f"SED fitting results for {repr(self.SED_fitter)} " + \
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
            
    @staticmethod
    def shorten_kwarg_colname(
        kwarg_colname: str,
        max_len: int = 68,
    ) -> str:
        """Shorten a selection kwarg column name to fit FITS limits.

        Strips JWST/HST filter prefixes/suffixes (e.g. `F444W` -> `444`)
        from the column name if it exceeds `max_len` characters, since
        FITS binary table column names have a fixed length limit.

        Parameters
        ----------
        kwarg_colname : `str`
            The full, unshortened column name.
        max_len : `int`, optional
            Maximum permitted column name length. Default is `68`.

        Returns
        -------
        `str`
            The (possibly shortened) column name.

        Raises
        ------
        AssertionError
            If the shortened name still exceeds `max_len` characters.
        """
        import re
        if len(kwarg_colname) > max_len:
            galfind_logger.debug(
                f"Shortening {kwarg_colname=} to avoid FITS column name length limits" + \
                "by removing band prefix/suffixes!" 
            )
            # remove any FXXXW/LP/M filter prefixes/suffixes from property name to shorten it
            save_property_name = re.sub(r'F(\d+)(?:LP|[WM])', r'\1', kwarg_colname)
        else:
            save_property_name = kwarg_colname
        assert len(save_property_name) <= max_len, \
            galfind_logger.critical(
                f"{len(save_property_name)=}>{max_len}. Please shorten '{save_property_name}'" + \
                "to avoid FITS column name length limits."
            )
        return save_property_name
        

class Data_Selector(Selector, ABC):
    """Abstract mixin for selectors acting on `Data`/catalogue-level metadata.

    Used for criteria that do not require per-band photometry, SED
    fitting, or morphology fitting to be loaded (e.g. ID cuts, region
    masks, unmasked-band counts). `_check_phot_exists`,
    `_check_SED_fit_exists`, and `_check_morph_fit_exists` all trivially
    return `True`, so `_selection_criteria` is always evaluated.

    Parameters
    ----------
    **kwargs
        Selector-specific keyword arguments; must include all names
        listed in `_include_kwargs`.
    """

    def __init__(self: Self, **kwargs) -> Self:
        super().__init__(
            aper_diam = None,
            SED_fitter = None,
            morph_fitter = None,
            **kwargs,
        )

    @property
    def requires_phot(self: Self) -> bool:
        """`bool`: Whether this selector requires photometry to be loaded (always `True`)."""
        return True

    @property
    def name(self: Self) -> str:
        """`str`: Unique name of this selection, equal to `_selection_name`."""
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
        """Apply this data-level selection criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.
        *args, **kwargs
            Forwarded to `Selector.__call__`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.
        """
        return Selector.__call__(self, object, return_copy, *args, **kwargs)


class Photometry_Selector(Selector, ABC):
    """Abstract mixin for selectors acting on aperture photometry.

    Used for criteria evaluated directly from the aperture photometry
    of a galaxy (e.g. band magnitude/SNR cuts, colour cuts) at a given
    aperture diameter. Requires photometry to exist at `aper_diam` but
    not SED-fitting or morphology-fitting results.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter (an angular size, e.g. in arcsec) at which the
        photometry used by this selector was extracted. Must be
        positive.
    **kwargs
        Selector-specific keyword arguments; must include all names
        listed in `_include_kwargs`.

    Raises
    ------
    AssertionError
        If `aper_diam` is not an angular `astropy.units.Quantity` or is
        not positive.
    """

    def __init__(self: Self, aper_diam: u.Quantity, **kwargs) -> Self:
        assert isinstance(aper_diam, u.Quantity)
        assert aper_diam.unit.is_equivalent(u.arcsec)
        assert aper_diam > 0 * u.arcsec
        super().__init__(aper_diam, SED_fitter = None, morph_fitter = None, **kwargs)

    @property
    def name(self: Self) -> str:
        """`str`: Unique name of this selection, including the aperture diameter."""
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
        """Apply this photometry-based selection criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.
        """
        return Selector.__call__(self, object, return_copy)


class SED_fit_Selector(Selector, ABC):
    """Abstract mixin for selectors acting on SED-fitting results.

    Used for criteria evaluated from the results of a given SED-fitting
    code/run (e.g. redshift limits, chi-squared cuts, rest-frame
    property cuts) at a given aperture diameter. Requires photometry and
    SED-fitting results for `SED_fitter` to exist, but not morphology
    fitting.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter at which the photometry underlying the SED
        fit was extracted. Must be positive.
    SED_fitter : `SED_code`
        The SED-fitting code/run whose results this selector acts on.
    **kwargs
        Selector-specific keyword arguments; must include all names
        listed in `_include_kwargs`.

    Raises
    ------
    AssertionError
        If `aper_diam` is not a positive angular `astropy.units.Quantity`,
        or `SED_fitter` is not an `SED_code` instance.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        **kwargs
    ) -> Self:
        assert isinstance(aper_diam, u.Quantity)
        assert aper_diam.unit.is_equivalent(u.arcsec)
        assert aper_diam > 0 * u.arcsec
        assert isinstance(SED_fitter, funcs.all_subclasses(SED_code)), \
            galfind_logger.critical(
                f"{repr(SED_fitter)} must be an SED_code object."
            )
        Selector.__init__(self, aper_diam, SED_fitter, morph_fitter = None, **kwargs)

    @property
    def requires_SED_fit(self: Self) -> bool:
        """`bool`: Whether this selector requires SED-fitting results (`True`)."""
        return True

    @property
    def name(self: Self) -> str:
        """`str`: Unique name of this selection, including SED fitter label and aperture diameter."""
        return f"{self._selection_name}_{self.SED_fitter.label.replace('_zfree', '')}" + \
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
            passed = self.SED_fitter.label in gal.aper_phot[self.aper_diam].SED_results.keys()
        except:
            passed = False
        return passed
    
    def _check_morph_fit_exists(
        self: Self,
        *args,
        **kwargs,
    ) -> bool:
        return True
    
    def _assert_SED_fitter(
        self: Self,
        object: Union[Galaxy, Type[Catalogue_Base]],
    ) -> str:
        if isinstance(object, Galaxy):
            assert self.SED_fitter.label in object.aper_phot[self.aper_diam].SED_results.keys(), \
                galfind_logger.critical(
                    f"SED fitting results for {repr(self.SED_fitter)} " + \
                    f"not loaded for {repr(object)}!"
                )
        elif isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            if not all(self.SED_fitter.label in gal.aper_phot[self.aper_diam].SED_results.keys() for gal in object):
                self.SED_fitter(object, self.aper_diam, update = True)
                # galfind_logger.critical(
                #     f"Performing {repr(self.SED_fitter)} on {repr(object)}."
                # )
        else:
            raise ValueError(
                f"{object=} with {type(object)=} not in ['Galaxy', 'Catalogue']!"
            )
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Union[Galaxy, Catalogue]]:
        """Apply this SED-fit-based selection criterion.

        Asserts that SED-fitting results for `self.SED_fitter` are
        loaded (running the fit on the catalogue first if necessary)
        before evaluating the criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.
        *args, **kwargs
            Forwarded to `Selector.__call__`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.
        """
        self._assert_SED_fitter(object)
        return Selector.__call__(self, object, return_copy, *args, **kwargs)


class Morphology_Selector(Selector, ABC):
    """Abstract mixin for selectors acting on morphology-fitting results.

    Used for criteria evaluated from the results of a given
    morphology-fitting code/run (e.g. effective radius cuts). Requires
    morphology-fitting results for `morph_fitter` to exist for the
    relevant cutout, but not photometry or SED fitting.

    Parameters
    ----------
    morph_fitter : `str` or `Morphology_Fitter`
        The morphology-fitting code/run whose results this selector
        acts on.
    **kwargs
        Selector-specific keyword arguments; must include all names
        listed in `_include_kwargs`.

    Raises
    ------
    AssertionError
        If `morph_fitter` is not a `Morphology_Fitter` subclass instance.
    """

    def __init__(
        self: Self,
        morph_fitter: Union[str, Morphology_Fitter],
        **kwargs,
    ) -> Self:
        from . import Morphology_Fitter
        assert isinstance(morph_fitter, tuple(Morphology_Fitter.__subclasses__()))
        super().__init__(aper_diam = None, SED_fitter = None, morph_fitter = morph_fitter, **kwargs)

    @property
    def requires_SED_fit(self: Self) -> bool:
        """`bool`: Whether this selector requires SED-fitting results (always `False`)."""
        return False

    @property
    def name(self: Self) -> str:
        """`str`: Unique name of this selection, including the morphology fitter name."""
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
        """Apply this morphology-based selection criterion.

        Asserts that morphology-fitting results for `self.morph_fitter`
        are loaded before evaluating the criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.

        Raises
        ------
        AssertionError
            If morphology-fitting results are not loaded for `object`
            (or for any galaxy in it, if a catalogue).
        """
        self._assert_morph_fitter(object)
        return Selector.__call__(self, object, return_copy)


class Redshift_Selector(SED_fit_Selector, ABC):
    """Abstract mixin for selectors acting on best-fit/rest-frame redshift quantities.

    Specializes `SED_fit_Selector` for criteria that do not themselves
    require additional SED-fit result loading beyond what
    `SED_fit_Selector` already checks (e.g. redshift limits,
    Lyman-break/Lyman-limit non-detection criteria).
    """

    @property
    def requires_SED_fit(self: Self) -> bool:
        """`bool`: Whether this selector requires SED-fitting results (always `False`)."""
        return False


class Mask_Selector(Data_Selector, ABC):
    """Abstract mixin for selectors that apply a spatial/data mask.

    Subclasses must implement `load_mask`, which loads or constructs the
    boolean sky mask for a given `Data` object.
    """

    @abstractmethod
    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load or construct the mask associated with this selector.

        Parameters
        ----------
        data : `Data`
            The dataset to load/construct the mask for.
        invert : `bool`, optional
            If `True`, invert the mask before returning it. Default is
            `True`.
        **kwargs
            Additional implementation-specific keyword arguments.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            The (optionally inverted) mask.
        """
        pass

    def extract_zbins(
        self: Self,
        data: Data,
    ) -> Optional[List[float]]:
        """Extract redshift bin edges associated with this mask, if any.

        Parameters
        ----------
        data : `Data`
            The dataset to extract redshift bins for.

        Returns
        -------
        `list` of `float` or `None`
            The redshift bin edges, or `None` if this selector's mask is
            not redshift-dependent. The base implementation always
            returns `None`.
        """
        return None


class Multiple_Selector(ABC):
    """Abstract mixin combining multiple `Selector` instances into one.

    Applies each selector in `selectors` in turn and requires all of
    them to pass (`_selection_criteria` is a logical AND over the
    sub-selectors) for a galaxy to be selected; a galaxy fails early
    (`_failure_criteria`) if any sub-selector's failure criteria is met.
    Concrete `Multiple_*_Selector` subclasses combine this mixin with a
    specific `Selector` mixin (`Data_Selector`, `Photometry_Selector`,
    `SED_fit_Selector`, `Mask_Selector`) to also satisfy that mixin's
    interface.

    Parameters
    ----------
    selectors : `list` of `Selector`
        The individual selectors to combine.
    selection_name : `str`, optional
        Name to use for the combined selection. If not given, defaults
        to the `_selection_name` of each sub-selector joined by `'+'`.

    Attributes
    ----------
    selectors : `numpy.ndarray`
        Flattened array of the combined `Selector` instances.
    selection_name : `str`
        Name of the combined selection, if explicitly given.
    """

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
        *args,
        **kwargs,
    ) -> bool:
        return any(
            selector._failure_criteria(gal, *args, **kwargs)
            for selector in self.selectors
        )

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        return all(
            selector._selection_criteria(gal, *args, **kwargs)[0]
            for selector in self.selectors
        ), {}


class Multiple_Data_Selector(Multiple_Selector, Data_Selector, ABC):
    """Abstract combination of multiple `Data_Selector` criteria.

    Applies each sub-selector's data-level cut in turn, then requires
    all to pass. Each sub-selector's `kwargs["filt_name"]` (if present)
    is checked against the target filterset via `crop_to_filterset`.

    Parameters
    ----------
    selectors : `list` of `Selector`
        The individual `Data_Selector` instances to combine.
    selection_name : `str`, optional
        Name to use for the combined selection. Default is `None`.
    fit_filterset : `Multiple_Filter`, optional
        If given, immediately crop `selectors` to only those whose
        `filt_name` kwarg is in this filterset. Default is `None`.
    """

    def __init__(
        self: Self,
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None,
        fit_filterset: Optional[Multiple_Filter] = None,
    ):
        Multiple_Selector.__init__(self, selectors, selection_name)
        Data_Selector.__init__(self, fit_filterset = fit_filterset)
        if fit_filterset is not None:
            self.crop_to_filterset(fit_filterset)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Type[Catalogue_Base]]]:
        """Apply each sub-selector in turn, then the combined criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.
        *args, **kwargs
            Forwarded to each sub-selector and to `Data_Selector.__call__`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.
        """
        # run selection individually on each selector
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            self.crop_to_filterset(object.filterset)
        [selector.__call__(object, return_copy = False) for selector in self.selectors]
        return Data_Selector.__call__(self, object, return_copy = return_copy)

    def crop_to_filterset(self: Self, filterset: Multiple_Filter) -> None:
        """Restrict the combined sub-selectors to those in a filterset.

        Parameters
        ----------
        filterset : `Multiple_Filter`
            The filterset to crop against. Sub-selectors whose
            `kwargs["filt_name"]` is not among `filterset.filt_names`
            are dropped.

        Returns
        -------
        `None`
            `self.selectors` is updated in place.
        """
        # crop each selector to the filterset
        self.selectors = [
            selector for selector in self.selectors
            if selector.kwargs["filt_name"] in filterset.filt_names
        ]


class Multiple_Photometry_Selector(Multiple_Selector, Photometry_Selector, ABC):
    """Abstract combination of multiple `Photometry_Selector` criteria.

    Applies each sub-selector's photometry-based cut in turn (all at a
    common aperture diameter), then requires all to pass.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Common aperture diameter shared by this selector and all
        sub-selectors.
    selectors : `list` of `Selector`
        The individual `Photometry_Selector` instances to combine.
    selection_name : `str`, optional
        Name to use for the combined selection. Default is `None`.
    **kwargs
        Additional keyword arguments forwarded to `Multiple_Selector`
        and `Photometry_Selector`.

    Raises
    ------
    AssertionError
        If any sub-selector has an `aper_diam` other than `None` or
        matching `aper_diam`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None,
        **kwargs,
    ):
        assert all([selector.aper_diam == aper_diam or selector.aper_diam is None for selector in selectors])
        Multiple_Selector.__init__(self, selectors, selection_name, **kwargs)
        Photometry_Selector.__init__(self, aper_diam, **kwargs)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Union[Galaxy, Catalogue]]:
        """Apply each sub-selector in turn, then the combined criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.
        *args, **kwargs
            Forwarded to each sub-selector and to
            `Photometry_Selector.__call__`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.
        """
        # run selection individually on each selector
        [selector.__call__(object, return_copy = False, *args, **kwargs) for selector in self.selectors]
        return Photometry_Selector.__call__(self, object, return_copy = return_copy, *args, **kwargs)


class Multiple_SED_fit_Selector(Multiple_Selector, SED_fit_Selector, ABC):
    """Abstract combination of multiple `SED_fit_Selector` criteria.

    Applies each sub-selector's SED-fit-based cut in turn (all sharing a
    common aperture diameter and SED fitter), then requires all to pass.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Common aperture diameter shared by this selector and all
        sub-selectors.
    SED_fitter : `SED_code`
        Common SED-fitting code/run shared by this selector and all
        sub-selectors.
    selectors : `list` of `Selector`
        The individual `SED_fit_Selector` instances to combine.
    selection_name : `str`, optional
        Name to use for the combined selection. Default is `None`.

    Raises
    ------
    AssertionError
        If any sub-selector has an `aper_diam` or `SED_fitter` that is
        not `None` and does not match `aper_diam`/`SED_fitter`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None
    ):
        assert all([selector.aper_diam is None or selector.aper_diam == aper_diam for selector in selectors])
        assert all([selector.SED_fitter is None or selector.SED_fitter.label == SED_fitter.label for selector in selectors])
        Multiple_Selector.__init__(self, selectors, selection_name)
        SED_fit_Selector.__init__(self, aper_diam, SED_fitter)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Catalogue]]:
        """Apply each sub-selector in turn, then the combined criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.
        *args, **kwargs
            Forwarded to each sub-selector and to
            `SED_fit_Selector.__call__`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.
        """
        # run selection individually on each selector
        [
            selector(object, return_copy = False, *args, **kwargs)
            for selector in self.selectors
        ]
        return SED_fit_Selector.__call__(
            self,
            object,
            return_copy = return_copy,
            *args,
            **kwargs,
        )


class Multiple_Mask_Selector(Multiple_Selector, Mask_Selector, ABC):
    """Abstract combination of multiple `Mask_Selector` criteria.

    Applies each sub-selector's mask-based cut in turn, then requires
    all to pass; the combined mask (`load_mask`) is the logical AND of
    each sub-selector's mask.

    Parameters
    ----------
    selectors : `list` of `Mask_Selector`
        The individual `Mask_Selector` instances to combine.
    selection_name : `str`, optional
        Name to use for the combined selection. Default is `None`.
    fit_filterset : `Multiple_Filter`, optional
        If given, immediately crop `selectors` to only those relevant
        to this filterset. Default is `None`.

    Raises
    ------
    AssertionError
        If any element of `selectors` is not a `Mask_Selector` instance.
    """

    def __init__(
        self: Self,
        selectors: List[Type[Mask_Selector]],
        selection_name: Optional[str] = None,
        fit_filterset: Optional[Multiple_Filter] = None,
    ):
        assert all([isinstance(selector, Mask_Selector) for selector in selectors])
        Multiple_Selector.__init__(self, selectors, selection_name)
        Mask_Selector.__init__(self, fit_filterset = fit_filterset)
        # NOT SURE IF THIS IS REQUIRED?
        if fit_filterset is not None:
            self.crop_to_filterset(fit_filterset)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Type[Catalogue_Base]]]:
        """Apply each sub-selector in turn, then the combined criterion.

        Parameters
        ----------
        object : `Galaxy` or `Catalogue`
            The galaxy or catalogue to select on.
        return_copy : `bool`, optional
            If `True`, operate on and return a deep copy. Default is `True`.
        *args, **kwargs
            Forwarded to each sub-selector and to `Mask_Selector.__call__`.

        Returns
        -------
        `Galaxy`, `Catalogue`, or `None`
            The annotated (and possibly copied/cropped) object.
        """
        # NOT SURE IF THIS IS REQUIRED?
        # run selection individually on each selector
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            self.crop_to_filterset(object.filterset)
        [selector.__call__(object, return_copy = False) for selector in self.selectors]
        return Mask_Selector.__call__(self, object, return_copy = return_copy)

    # NOT SURE IF THIS IS REQUIRED?
    def crop_to_filterset(self: Self, filterset: Multiple_Filter) -> None:
        """Restrict the combined sub-selectors to those relevant to a filterset.

        Parameters
        ----------
        filterset : `Multiple_Filter`
            The filterset to crop against. Sub-selectors with a
            `filt_name` kwarg not in `filterset.filt_names` are
            dropped; sub-selectors without a `filt_name` kwarg are kept.

        Returns
        -------
        `None`
            `self.selectors` is updated in place.
        """
        # crop each selector to the filterset
        selectors_arr = []
        for selector in self.selectors:
            append = False
            if "filt_name" in selector.kwargs.keys():
                if selector.kwargs["filt_name"] in filterset.filt_names:
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
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load the combined mask from all sub-selectors.

        Parameters
        ----------
        data : `Data`
            The dataset to load/construct the mask for.
        invert : `bool`, optional
            If `True`, invert the combined mask before returning it.
            Default is `True`.
        **kwargs
            Additional keyword arguments forwarded to each
            sub-selector's `load_mask`.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            The logical AND of each sub-selector's (non-inverted) mask,
            optionally inverted.
        """
        # load mask from each selector
        masks = [selector.load_mask(data, False, **kwargs) for selector in self.selectors]
        # combine masks
        combined_mask = np.logical_and.reduce(masks)
        if invert:
            combined_mask = np.invert(combined_mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return combined_mask

    def extract_zbins(
        self: Self,
        data: Data,
    ) -> Optional[List[float]]:
        """Extract combined redshift bin edges from all sub-selectors.

        Parameters
        ----------
        data : `Data`
            The dataset to extract redshift bins for.

        Returns
        -------
        `list` of `list` of `float`
            Consecutive `[z_low, z_high]` pairs built from the sorted,
            unique union of each sub-selector's redshift bin edges.
        """
        zbins_arr = [selector.extract_zbins(data) for selector in self]
        zlims = np.unique(np.array([zbins for zbins in zbins_arr if zbins is not None]).flatten())
        return [[a, b] for a, b in zip(zlims, zlims[1:])]


class ID_Selector(Data_Selector):
    """Select individual galaxies by ID.

    Selects only the galaxies whose `Galaxy.ID` is in the given `IDs`
    list, optionally attaching arbitrary per-ID diagnostic keyword
    values (e.g. literature classifications) to the output table.

    Parameters
    ----------
    IDs : `int` or `list` of `int`
        The galaxy ID(s) to select. A single `int` is wrapped in a
        list.
    name : `str`, optional
        Name to use for this selection. If `None`, defaults to
        ``f"ID_{IDs}"``. Default is `None`.
    **select_kwargs : `dict`
        Additional per-ID diagnostic keyword arguments; each value
        must be a sequence the same length as `IDs`, giving the value
        to store for the corresponding ID.
    """

    def __init__(
        self: Self,
        IDs: Union[int, List[int]],
        name: Optional[str] = None,
        **select_kwargs: Dict[str, Any],
    ):
        if isinstance(IDs, int):
            IDs = [IDs]
        kwargs = {"IDs": IDs, "name": name, "select_kwargs": select_kwargs}
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        if self.kwargs["name"] is None:
            return f"ID_{self.kwargs['IDs']}"
        else:
            return self.kwargs["name"]

    @property
    def _include_kwargs(self) -> List[str]:
        return ["IDs", "name", "select_kwargs"]

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names and `str` dtypes of the `select_kwargs`.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            The keys of `self.kwargs["select_kwargs"]` and a `str`
            dtype for each.
        """
        return [key for key in self.kwargs["select_kwargs"].keys()], [str] * len(self.kwargs["select_kwargs"])

    def _assertions(self: Self) -> bool:
        try:
            assert isinstance(self.kwargs["IDs"], tuple([list, np.ndarray]))
            assert all([id >= 1 for id in self.kwargs["IDs"]])
            if self.kwargs["name"] is not None:
                assert isinstance(self.kwargs["name"], str) 
            assert all(np.isscalar(ID) and np.issubdtype(type(ID), np.integer) for ID in self.kwargs["IDs"])
            assert all([len(self.kwargs["IDs"]) == len(kwarg_vals) for kwarg_vals in self.kwargs["select_kwargs"].values()]), \
                galfind_logger.critical(
                    f"All {self.kwargs['IDs']=} and " + \
                    f"{len([vals for vals in self.kwargs['select_kwargs'].values()])}" + \
                    f" must have the same length!"
                )
            passed = True
        except:
            passed = False
        return passed
    
    def _assert_cat(self: Self, cat: Catalogue) -> None:
        cat_IDs = np.array([gal.ID for gal in cat])
        assert all(ID in cat_IDs for ID in self.kwargs["IDs"]), \
            galfind_logger.critical(
                f"Not all {self.kwargs['IDs']=} found in {repr(cat)} with {cat_IDs=}!"
            )
        super()._assert_cat(cat)
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        out_kwargs = {}
        for kwarg_name, kwarg_values in self.kwargs["select_kwargs"].items():
            if gal.ID not in self.kwargs["IDs"]:
                out_kwargs[kwarg_name] = ""
            else:
                out_kwargs[kwarg_name] = kwarg_values[np.where(np.array(self.kwargs["IDs"]) == gal.ID)[0][0]]
        return gal.ID in self.kwargs["IDs"], out_kwargs


class Region_Selector(Data_Selector, ABC):
    """Abstract mixin for selectors defined by a spatial region mask.

    Builds (via `make_mask`, implemented by subclasses), caches to
    disk, and applies a boolean region mask as a selection criterion.
    Selected galaxies are tagged with `name`; unselected galaxies are
    tagged with `fail_name` in `Galaxy.region`/`Catalogue.regions`.

    Parameters
    ----------
    fail_name : `str`, optional
        Name to use for galaxies failing this region selection. If
        `None`, defaults to ``f"not_{self._selection_name}"``. Default
        is `None`.
    **kwargs
        Additional keyword arguments forwarded to
        `Data_Selector.__init__`.
    """

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
        """Construct the boolean region mask for a catalogue's data.

        Parameters
        ----------
        cat : `Catalogue`
            The catalogue (and associated `Data`) to construct the
            mask for.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            Boolean mask array, `True` where selected.
        """
        pass

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load this region's previously-saved mask from disk.

        Parameters
        ----------
        data : `Data`
            The dataset whose region mask FITS file (at
            `get_mask_path`) should be loaded.
        invert : `bool`, optional
            If `True`, invert the mask before returning it. Default is
            `True`.
        **kwargs
            Unused; present for interface compatibility with other
            `Mask_Selector`-like `load_mask` implementations.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            The (optionally inverted) boolean region mask.
        """
        mask = fits.open(
            self.get_mask_path(data),
            mode = "readonly",
            ignore_missing_simple = True
        )[self.name].data.astype(bool)
        if invert:
            mask = np.logical_not(mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return mask

    @property
    def name(self: Self) -> str:
        """`str`: Unique name of this region selection."""
        return self._selection_name

    @property
    def fail_name(self: Self) -> str:
        """`str`: Name used to tag galaxies failing this region selection.

        Returns
        -------
        `str`
            The `fail_name` given at construction, if set; otherwise
            ``f"not_{self._selection_name}"``.

        Raises
        ------
        AssertionError
            If an explicitly set `fail_name` is equal to `name`.
        """
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
        """Determine the path this region's cached mask FITS file is stored at.

        Parameters
        ----------
        data : `Data`
            The dataset the mask is associated with, used to build the
            survey/version-specific path.

        Returns
        -------
        `str`
            Path to the region mask FITS file under
            ``config['Masking']['MASK_DIR']``.
        """
        return f"{config['Masking']['MASK_DIR']}/{data.survey}/region_masks/{data.version}_{self.name}.fits"

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Catalogue]:
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
            if name not in cat.regions:
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
    ) -> Optional[Galaxy]:
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
    """Region selector defined by a DS9 ``.reg`` region file.

    Notes
    -----
    This selector is not yet fully implemented: `make_mask` and
    `_selection_criteria` both currently raise `Exception`.

    Parameters
    ----------
    region_path : `str`
        Path to the DS9 ``.reg`` region file.
    region_name : `str`, optional
        Name to use for this region. If `None`, derived from the
        filename in `region_path` (stripped of its ``.reg`` extension
        and leading path). Default is `None`.
    fail_name : `str`, optional
        Name to use for galaxies failing this region selection.
        Default is `None`.
    """

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
        """Not implemented.

        Raises
        ------
        Exception
            Always; mask construction from a DS9 region is not yet
            implemented.
        """
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
        with open(self.kwargs["region_path"], "r") as f:
            # save .reg file to object
            pass
        return super().__call__(object, return_copy, *args, **kwargs)

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        raise Exception()


class Depth_Region_Selector(Region_Selector):
    """Region selector splitting a catalogue by per-galaxy depth label.

    Assigns each galaxy to the nearest-neighbour-filled depth region
    (from `Depths.get_grid_depth_path`) computed for a given
    band/aperture, then selects galaxies whose depth region label
    matches `region_label`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the depths were computed at.
    filt_name : `str` or `list` of `str`
        Name(s) of the band(s) (joined by ``'+'`` if a list) the depth
        regions are computed for.
    region_label : `int` or `str`
        Label of the depth region to select.
    region_name : `str`, optional
        Name to use for this region. If `None`, derived from
        `filt_name` and `region_label`. Default is `None`.
    fail_name : `str`, optional
        Name to use for galaxies failing this region selection.
        Default is `None`.
    """

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

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic `depth_label` kwarg.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["depth_label"]` and `[str]`.
        """
        return ["depth_label"], [str]

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
    ) -> Optional[Catalogue]:
        # add array of depth regions to kwargs
        # TODO: Add these as options; hard coded for now!
        if all(self._selection_name in gal.selection_flags.keys() for gal in cat):
            galfind_logger.info(
                f"Depth regions already selected for {repr(cat)}!"
            )
        else:
            if self.kwargs['filt_name'] == cat.data.forced_phot_band.filt_name:
                band_data_base = cat.data.forced_phot_band
            else:
                band_data_base = cat.data[self.kwargs["filt_name"]]
            h5_path = Depths.get_grid_depth_path(
                band_data_base,
                self._aper_diam,
                mode = "n_nearest",
            )
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
        **kwargs,
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
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        kwargs_out = {}
        if "reg_dict" not in kwargs.keys():
            kwargs_out["depth_label"] = np.nan
            return True, kwargs_out
        depth_label = kwargs["reg_dict"][gal.ID]
        if isinstance(depth_label, float):
            depth_label = str(int(depth_label))
        kwargs_out["depth_label"] = depth_label
        if depth_label == self.kwargs["region_label"]:
            return True, kwargs_out
        else:
            return False, kwargs_out


class Redshift_Limit_Selector(Redshift_Selector):
    """Select galaxies above or below a best-fit redshift limit.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run whose best-fit redshift this selector
        acts on.
    z_lim : `float` or `int`
        Redshift limit to compare against.
    gtr_or_less : `str`
        Either ``"gtr"`` to select `z >= z_lim`, or ``"less"`` to
        select `z <= z_lim`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        z_lim: Union[float, int],
        gtr_or_less: str
    ):
        kwargs = {"z_lim": z_lim, "gtr_or_less": gtr_or_less}
        super().__init__(aper_diam, SED_fitter, **kwargs)

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
            assert self.kwargs["z_lim"] >= 0.0
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
    ) -> Tuple[bool, Dict[str, Any]]:
        if self.kwargs["gtr_or_less"] == "gtr":
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fitter.label].z >= self.kwargs["z_lim"], {}
        else:
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fitter.label].z <= self.kwargs["z_lim"], {}
        

class Rest_Frame_Property_Limit_Selector(Redshift_Selector):
    """Select galaxies above or below a limit on a rest-frame property.

    Computes `property_calculator` for the target object before
    applying the selection (see `__call__`).

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run the rest-frame property was derived from.
    property_calculator : `Type`[`Property_Calculator`]
        Calculator for the rest-frame property to select on.
    property_lim : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
        Property value limit to compare against.
    gtr_or_less : `str`
        Either ``"gtr"`` to select `property > property_lim`, or
        ``"less"`` to select `property < property_lim`.

    Attributes
    ----------
    property_calculator : `Type`[`Property_Calculator`]
        The rest-frame property calculator, as given at construction.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
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
        super().__init__(aper_diam, SED_fitter, **kwargs)

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
            assert self.SED_fitter.label == self.property_calculator.SED_fit_label
            passed = True
        except:
            passed = False
        return passed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        try:
            assertions = []
            assertions.extend([
                self.kwargs["property_name"] in gal.aper_phot[self.aper_diam].SED_results \
                    [self.SED_fitter.label].phot_rest.properties.keys()
            ])
            assertions.extend([
                gal.aper_phot[self.aper_diam].SED_results \
                    [self.SED_fitter.label].phot_rest.properties \
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
    ) -> Tuple[bool, Dict[str, Any]]:
        if self.kwargs["gtr_or_less"] == "gtr":
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fitter.label].phot_rest.properties \
                [self.kwargs["property_name"]] \
                > self.kwargs["property_lim"], {}
        else:
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fitter.label].phot_rest.properties \
                [self.kwargs["property_name"]] \
                < self.kwargs["property_lim"], {}
    
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
    """Select galaxies with a best-fit redshift within a bin.

    Combines two `Redshift_Limit_Selector` instances (`z >= z_bin[0]`
    and `z <= z_bin[1]`) via `Multiple_SED_fit_Selector`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run whose best-fit redshift this selector
        acts on.
    z_bin : `list` of `int` or `float`
        Two-element `[z_low, z_high]` redshift bin edges, with
        `z_low < z_high`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        z_bin: List[Union[int, float]],
    ):
        assert all(isinstance(z_lim, (int, float)) for z_lim in z_bin)
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        selection_name = f"{z_bin[0]:.2f}<z<{z_bin[1]:.2f}"
        selectors = [
            Redshift_Limit_Selector(aper_diam, SED_fitter, z_bin[0], "gtr"),
            Redshift_Limit_Selector(aper_diam, SED_fitter, z_bin[1], "less"),
        ]
        super().__init__(aper_diam, SED_fitter, selectors, selection_name)


class Rest_Frame_Property_Bin_Selector(Multiple_SED_fit_Selector):
    """Select galaxies with a rest-frame property within a bin.

    Combines two `Rest_Frame_Property_Limit_Selector` instances
    (`property > property_bin[0]` and `property < property_bin[1]`)
    via `Multiple_SED_fit_Selector`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run the rest-frame property was derived from.
    property_calculator : `Rest_Frame_Property_Calculator`
        Calculator for the rest-frame property to select on.
    property_bin : `list` of `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
        Two-element `[low, high]` property bin edges, with
        `low < high`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
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
                SED_fitter,
                property_calculator,
                property_bin[0],
                "gtr"
            ),
            Rest_Frame_Property_Limit_Selector(
                aper_diam,
                SED_fitter,
                property_calculator,
                property_bin[1],
                "less"
            )
        ]
        super().__init__(aper_diam, SED_fitter, selectors, selection_name)


class Colour_Selector(Photometry_Selector):
    """Select galaxies bluer or redder than a colour cut.

    Computes the AB magnitude colour between two bands and compares it
    to `colour_val`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry was extracted at.
    colour_bands : `str`, `list` of `str`, or `numpy.ndarray` of `str`
        The two bands (blue, red) making up the colour, either as a
        ``"band1-band2"`` string or a 2-element list/array.
    bluer_or_redder : `str`
        Either ``"bluer"`` to select `colour < colour_val`, or
        ``"redder"`` to select `colour > colour_val`.
    colour_val : `float`
        Colour limit (in AB magnitudes) to compare against.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        colour_bands: Union[str, List[str], NDArray[str]],
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

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic `colour` kwarg.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["colour"]` and `[float]`.
        """
        return ["colour"], [float]

    def _assertions(self: Self) -> bool:
        try:
            assert self.kwargs["bluer_or_redder"] in ["bluer", "redder"]
            assert isinstance(self.kwargs["colour_bands"], (list, np.ndarray))
            assert len(self.kwargs["colour_bands"]) == 2
            passed = True
        except:
            passed = False
        return passed
        
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        try:
            assertions = []
            assertions.extend([
                all(
                    colour in gal.aper_phot[self.aper_diam].filterset.filt_names
                    for colour in self.kwargs["colour_bands"]
                )
            ])
            # ensure bands are ordered blue -> red
            assertions.extend([
                np.where(np.array(gal.aper_phot[self.aper_diam].filterset.filt_names) == self.kwargs["colour_bands"][0])[0][0] \
                    < np.where(np.array(gal.aper_phot[self.aper_diam].filterset.filt_names) == self.kwargs["colour_bands"][1])[0][0]
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
    ) -> Tuple[bool, Dict[str, Any]]:
        band_indices = [
            int(np.where(np.array(gal.aper_phot[self.aper_diam].\
            filterset.filt_names) == filt_name)[0][0])
            for filt_name in self.kwargs["colour_bands"]
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
        selected = (colour < self.kwargs["colour_val"] and \
            self.kwargs["bluer_or_redder"] == "bluer") or \
            (colour > self.kwargs["colour_val"] and \
            self.kwargs["bluer_or_redder"] == "redder")
        kwargs_out = {"colour": colour}
        return selected, kwargs_out


class Compactness_Selector(Data_Selector):
    """Select galaxies by point-source compactness in a single band.

    Compares the ratio of flux within two circular apertures
    (`compare_radii`) centred on the source, optionally normalising by
    the same flux ratio measured on the band's PSF, to
    `compactness_lim`.

    Parameters
    ----------
    filt_name : `str`
        Name of the band the compactness is measured in.
    compare_radii : `astropy.units.Quantity`
        Two-element `[small, large]` aperture radii to compare the
        enclosed flux ratio of.
    compactness_lim : `int` or `float`
        Compactness (`large`/`small` flux ratio) limit. A galaxy is
        selected if its flux ratio is below this limit (or below the
        limit times the PSF's flux ratio, if `psf_normed`).
    psf_normed : `bool`, optional
        If `True`, normalise the galaxy's flux ratio by the band's PSF
        flux ratio before comparing to `compactness_lim`. Default is
        `True`.
    """

    def __init__(
        self: Self,
        filt_name: str,
        compare_radii: u.Quantity,
        compactness_lim: Union[int, float],
        psf_normed: bool = True,
    ):
        kwargs = {
            "filt_name": filt_name,
            "compare_radii": compare_radii,
            "compactness_lim": compactness_lim,
            "psf_normed": psf_normed,
        }
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        name = f"compact_{self.kwargs['filt_name']}_" + \
            f"r={self.kwargs['compare_radii'][1].to(u.arcsec).value:.2f}/" + \
            f"{self.kwargs['compare_radii'][0].to(u.arcsec).value:.2f}as" + \
            f"<{self.kwargs['compactness_lim']:.2f}"
        if self.kwargs["psf_normed"]:
            name += "*psf"
        return name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["filt_name", "compare_radii", "compactness_lim", "psf_normed"]

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic compactness kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            Names `["flux_ratio", "centre", "xoff", "yoff", "2dg_dist"]`
            (plus `"psf_flux_ratio"` if `psf_normed`) and their
            corresponding output dtypes.
        """
        names = ["flux_ratio", "centre", "xoff", "yoff", "2dg_dist"]
        dtypes = [float, str, float, float, float]
        if self.kwargs["psf_normed"]:
            names.append("psf_flux_ratio")
            dtypes.append(float)
        return names, dtypes

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["filt_name"], str)])
            assertions.extend([self.kwargs["filt_name"] in json.loads(config.get("Other", "ALL_BANDS"))])
            assertions.extend([isinstance(self.kwargs["compare_radii"], u.Quantity)])
            assertions.extend([radii > 0 * u.arcsec for radii in self.kwargs["compare_radii"]])
            assertions.extend([len(self.kwargs["compare_radii"]) == 2])
            assertions.extend([self.kwargs["compare_radii"][0] < self.kwargs["compare_radii"][1]])
            assertions.extend([isinstance(self.kwargs["compactness_lim"], (int, float))])
            assertions.extend([isinstance(self.kwargs["psf_normed"], bool)])
            passed = all(assertions)
        except:
            passed = False
        return passed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        try:
            failed = self.kwargs["filt_name"] not in \
                gal.aper_phot[list(gal.aper_phot.keys())[0]].filterset.filt_names
        except:
            failed = True
        return failed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        large_radii_as = self.kwargs["compare_radii"][1].to(u.arcsec).value
        cutout_label = f"{self.kwargs['filt_name']}_" + \
            f"{(large_radii_as * 3):.2f}as_native"
        if "band_data" in kwargs.keys() and (
                not hasattr(gal, "cutouts") or \
                cutout_label not in gal.cutouts.keys()
            ):
            gal.make_band_cutout(
                kwargs["band_data"],
                cutout_size = np.round(self.kwargs["compare_radii"][1] * 3, 2).to(u.arcsec),
                overwrite = False,
            )
        assert cutout_label in gal.cutouts.keys(), \
            galfind_logger.critical(
                f"{cutout_label=} not found in {gal.cutouts.keys()}"
            )
        cutout = gal.cutouts[cutout_label]
        flux_ratio, flux_ratio_kwargs = self._compute_flux_ratio(
            cutout,
            centre = True,
            mask_neighbours = True,
        )
        kwargs_out = {"flux_ratio": flux_ratio, **flux_ratio_kwargs}
        if self.kwargs["psf_normed"]:
            assert cutout.band_data.psf is not None, \
                galfind_logger.critical(
                    f"PSF not found for {cutout.band_data}!"
                )
            psf_flux_ratio = self._compute_flux_ratio(
                cutout.band_data.psf.cutout,
                centre = False,
                mask_neighbours = False,
            )[0]
            kwargs_out["psf_flux_ratio"] = psf_flux_ratio
            selected = flux_ratio < self.kwargs["compactness_lim"] * psf_flux_ratio
        else:
            selected = flux_ratio < self.kwargs["compactness_lim"]
        return selected, kwargs_out

    def _compute_flux_ratio(
        self: Self,
        cutout: Band_Cutout,
        centre: bool = True,
        mask_neighbours: bool = True,
    ) -> Tuple[float, Dict[str, Union[str, float]]]:
        pix_scale = cutout.band_data.pix_scale
        # open band cutout and compute the compactness
        sci = cutout.band_data.load_im()[0]
        if mask_neighbours and centre:
            seg_data = cutout.band_data.load_seg()[0]
            # determine cutout ID in the segmap
            nonzero_yx = np.argwhere(seg_data != 0)
            if len(nonzero_yx) != 0:
                distances = np.hypot(
                    nonzero_yx[:, 0] - int((seg_data.shape[0] // 2)),
                    nonzero_yx[:, 1] - int((seg_data.shape[1] // 2))
                )
                closest_yx = nonzero_yx[np.argmin(distances)]
                primary_id = seg_data[closest_yx[0], closest_yx[1]]
                # mask out neighbours
                mask = (seg_data != primary_id)
            else:
                mask = None
                centre = False
        else:
            mask = None

        kwargs = {}
        seg_x0 = sci.shape[1] / 2
        seg_y0 = sci.shape[0] / 2
        if centre:
            if np.sum(np.isfinite(sci[~mask])) >= 6:
                # find center of cutout using centroid_2dg
                try:
                    x0, y0 = centroid_2dg(sci, mask = mask)
                except Exception as e:
                    galfind_logger.warning(
                        f"Error occurred while computing centroid for {repr(cutout)}: {e}. " + \
                        "Using cutout center as aperture center instead."
                    )
                    fail = True
                    kwargs["2dg_dist"] = None
                else:
                    distance_from_seg_centre = np.hypot(x0 - seg_x0, y0 - seg_y0)
                    if distance_from_seg_centre >= 10.0:
                        galfind_logger.debug(
                            f"{repr(cutout)} centroid is {distance_from_seg_centre:.2f}>=10.0 pixels " +
                            "from cutout center! Using cutout center as aperture center instead."
                        )
                        fail = True
                    else:
                        fail = False
                        kwargs["centre"] = "centroid_2dg"
                    kwargs["2dg_dist"] = distance_from_seg_centre
            else:
                fail = True
                kwargs["2dg_dist"] = None
            if fail:
                x0 = seg_x0
                y0 = seg_y0
                kwargs["centre"] = "segmap"
        else:
            x0 = seg_x0
            y0 = seg_y0
            kwargs["2dg_dist"] = None
            kwargs["centre"] = "segmap"
        kwargs["xoff"] = x0 - seg_x0
        kwargs["yoff"] = y0 - seg_y0
        # make circular apertures with radii rad_1 and rad_2
        rad_pix = (self.kwargs["compare_radii"] / pix_scale).to(u.dimensionless_unscaled).value
        aper_1 = CircularAperture((x0, y0), r = rad_pix[0])
        aper_2 = CircularAperture((x0, y0), r = rad_pix[1])
        # compute flux in each aperture
        flux_1 = aper_1.do_photometry(sci)[0][0]
        flux_2 = aper_2.do_photometry(sci)[0][0]
        # compute flux ratio
        flux_ratio = flux_2 / flux_1
        return flux_ratio, kwargs

    def _assert_cat(self: Self, cat: Catalogue) -> None:
        if isinstance(self.kwargs["filt_name"], str):
            assert (self.kwargs["filt_name"] in cat.filterset.filt_names), \
                galfind_logger.critical(
                    f"{self.kwargs['filt_name']} not in {cat.filterset.filt_names}."
                )
        super()._assert_cat(cat)
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Catalogue]:
        kwargs["band_data"] = cat.data.native[self.kwargs["filt_name"]]
        return super()._call_cat(cat, return_copy, *args, **kwargs)


class Min_Band_Selector(Data_Selector):
    """Select galaxies detected in at least a minimum number of bands.

    Parameters
    ----------
    min_bands : `int`
        Minimum number of bands (with photometry present, regardless
        of masking) required for selection.
    """

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
    
    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic `n_bands` kwarg.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["n_bands"]` and `[int]`.
        """
        return ["n_bands"], [int]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["min_bands"], int)])
            assertions.extend([self.kwargs["min_bands"] > 0])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        n_bands = len(gal.aper_phot[list(gal.aper_phot.keys())[0]])
        return n_bands >= self.kwargs["min_bands"], {"n_bands": n_bands}


class Unmasked_Band_Selector(Mask_Selector):
    """Select galaxies unmasked in a single band.

    Parameters
    ----------
    filt_name : `str`
        Name of the band to check the mask of.
    """

    def __init__(
        self: Self,
        filt_name: str,
    ):
        kwargs = {"filt_name": filt_name}
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        return f"unmasked_{self.kwargs['filt_name']}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["filt_name"]

    def _assertions(self: Self) -> bool:
        # ensure that each band is a valid band name in galfind
        return self.kwargs["filt_name"] in json.loads(config.get("Other", "ALL_BANDS"))
        
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        try:
            failed = self.kwargs["filt_name"] not in \
                gal.aper_phot[list(gal.aper_phot.keys())[0]].filterset.filt_names
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        band_index = int([i for i, filt_name in enumerate( \
            gal.aper_phot[list(gal.aper_phot.keys())[0]].filterset.filt_names) \
            if filt_name == self.kwargs["filt_name"]][0])
        if not isinstance(gal.aper_phot[list(gal.aper_phot.keys())[0]].flux, Masked):
            return True, {}
        else:
            return not gal.aper_phot[list(gal.aper_phot.keys())[0]].flux.mask[band_index], {}
    
    def _assert_cat(self: Self, cat: Catalogue) -> None:
        assert self.kwargs["filt_name"] in cat.filterset.filt_names, \
            galfind_logger.critical(
                f"{self.kwargs['filt_name']} not in {cat.filterset.filt_names}."
            )
        super()._assert_cat(cat)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Catalogue]:
        return Data_Selector._call_cat(self, cat, return_copy, *args, **kwargs)

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load this band's mask from a `Data` object.

        Parameters
        ----------
        data : `Data`
            The dataset to load the band mask from. If the band is not
            present in `data.filterset`, an all-`False` mask matching
            the shape of the other bands is returned instead (with a
            warning logged).
        invert : `bool`, optional
            If `True`, invert the mask before returning it. Default is
            `True`.
        **kwargs
            Unused; present for interface compatibility.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            The (optionally inverted) boolean band mask.

        Raises
        ------
        AssertionError
            If the data shapes of the bands in `data` do not all
            match (only checked when `filt_name` is not in
            `data.filterset`).
        """
        # load mask from each selector
        if self.kwargs["filt_name"] not in data.filterset.filt_names:
            data_shapes = [band_data.data_shape for band_data in data]
            assert all(data_shape == data_shapes[0] for data_shape in data_shapes), \
                galfind_logger.critical(
                    f"Data shapes do not match: {data_shapes}"
                )
            mask = np.full(data_shapes[0], False)
            galfind_logger.warning(
                f"{self.kwargs['filt_name']} not in {data.filterset.filt_names}!"
            )
        else:
            # extract band_data from data
            band_data = data[self.kwargs["filt_name"]]
            # load mask from data
            mask = band_data.load_mask("MASK")[0].astype(bool)
        if not invert:
            mask = np.logical_not(mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return mask


class Min_Unmasked_Band_Selector(Mask_Selector):
    """Select galaxies unmasked in at least a minimum number of bands.

    Parameters
    ----------
    min_bands : `int`
        Minimum number of unmasked bands required for selection.
    """

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

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic `n_unmasked_bands` kwarg.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["n_unmasked_bands"]` and `[int]`.
        """
        return ["n_unmasked_bands"], [int]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["min_bands"], int)])
            assertions.extend([self.kwargs["min_bands"] > 0])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        if isinstance(gal.aper_phot[list(gal.aper_phot.keys())[0]].flux, u.Quantity):
            mask = np.full(len(gal.aper_phot[list(gal.aper_phot.keys())[0]].flux), False)
        else:
            mask = gal.aper_phot[list(gal.aper_phot.keys())[0]].flux.mask
        n_unmasked_bands = len([val for val in mask if not val])
        return n_unmasked_bands >= self.kwargs["min_bands"], {"n_unmasked_bands": n_unmasked_bands}

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load the combined "at least `min_bands` unmasked" mask.

        Parameters
        ----------
        data : `Data`
            The dataset to construct the mask from; the number of
            unmasked bands is summed pixel-by-pixel across all bands
            in `data`.
        invert : `bool`, optional
            If `True`, invert the mask before returning it. Default is
            `True`.
        **kwargs
            Unused; present for interface compatibility.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            Boolean mask, `True` where at least `min_bands` bands are
            unmasked (before any `invert`).

        Raises
        ------
        AssertionError
            If the data shapes of the bands in `data` do not all
            match.
        """
        # add masks
        data_shapes = [band_data.data_shape for band_data in data]
        assert all(data_shape == data_shapes[0] for data_shape in data_shapes), \
            galfind_logger.critical(
                f"Data shapes do not match: {data_shapes}"
            )
        n_bands_unmasked = np.zeros(data_shapes[0])
        for band_data in data:
            n_bands_unmasked += band_data.load_mask()
        # convert to boolean (True if n_bands_unmasked >= min_bands else False)
        mask = n_bands_unmasked >= self.kwargs["min_bands"]
        mask = mask.astype(bool)
        if invert:
            mask = np.logical_not(mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return mask


class Min_Instrument_Unmasked_Band_Selector(Mask_Selector):
    """Select galaxies unmasked in a minimum number of bands of one instrument.

    Parameters
    ----------
    min_bands : `int`
        Minimum number of unmasked bands (restricted to `instrument`)
        required for selection.
    instrument : `str` or `Type`[`Instrument`]
        The instrument (e.g. ``"NIRCam"``) to count unmasked bands of.
        If a `str`, it is converted to the corresponding `Instrument`
        subclass instance.
    """

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

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic `n_unmasked_bands` kwarg.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["n_unmasked_bands"]` and `[int]`.
        """
        return ["n_unmasked_bands"], [int]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["min_bands"], int)])
            assertions.extend([self.kwargs["min_bands"] > 0])
            assertions.extend([isinstance(self.kwargs["instrument"], tuple(Instrument.__subclasses__()))])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
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
        return n_unmasked_bands >= self.kwargs["min_bands"], {"n_unmasked_bands": n_unmasked_bands}

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load the combined "at least `min_bands` unmasked" mask for one instrument.

        Parameters
        ----------
        data : `Data`
            The dataset to construct the mask from; the number of
            unmasked bands is summed pixel-by-pixel across bands in
            `data` belonging to `instrument`.
        invert : `bool`, optional
            If `True`, invert the mask before returning it. Default is
            `True`.
        **kwargs
            Unused; present for interface compatibility.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            Boolean mask, `True` where at least `min_bands` bands of
            `instrument` are unmasked (before any `invert`).

        Raises
        ------
        AssertionError
            If the data shapes of the bands in `data` do not all
            match.
        """
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
    """Select galaxies non-detected in all bands bluewards of the Lyman limit.

    Requires the first band entirely bluewards of the (redshifted)
    Lyman limit, and all bands further bluewards, to have
    `SNR < SNR_lim` or be masked. Galaxies with no bands bluewards of
    the Lyman limit are not selected.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry/SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run providing the best-fit redshift used to
        place the Lyman limit.
    SNR_lim : `float`
        Signal-to-noise ratio non-detection threshold.
    fit_filterset : `Multiple_Filter`, optional
        If given, restrict the bands considered to those in this
        filterset. Default is `None`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        SNR_lim: float,
        fit_filterset: Optional[Multiple_Filter] = None,
    ):
        kwargs = {"SNR_lim": SNR_lim}
        super().__init__(aper_diam, SED_fitter, fit_filterset = fit_filterset, **kwargs)

    @property
    def _selection_name(self) -> str:
        selection_name = f"blue_LyLim<{self.kwargs['SNR_lim']:.1f}"
        # if self.kwargs["ignore_bands"] is not None:
        #     ignore_str = ",".join(self.kwargs["ignore_bands"])
        #     selection_name += f"_no_{ignore_str}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim"]#, "ignore_bands"]

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic non-detection kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            Names `["1st_filt", "SNR", "mask"]` (the first Lyman-limit
            non-detect band, comma-separated SNRs, and comma-separated
            mask flags of the bands checked) and their `str` dtypes.
        """
        return ["1st_filt", "SNR", "mask"], [str, str, str]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            # if self.kwargs["ignore_bands"] is not None:
            #     for band in self.kwargs["ignore_bands"]:
            #         # ensure this band exists
            #         assertions.extend([band in json.loads(config.get("Other", "ALL_BANDS"))])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        # extract first Lylim non-detect band
        if self.fit_filterset is None:
            ignore_bands = []
        else:
            ignore_bands = [
                filt_name for filt_name in 
                gal.aper_phot[self.aper_diam].filterset.filt_names
                if filt_name not in self.fit_filterset.filt_names
            ]
        first_Lylim_non_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fitter.label].phot_rest. \
                get_first_bluewards_band(
                    wav_lyman_lim * u.AA,
                    ignore_bands,
                )
        kwargs_out = {
            "1st_filt": first_Lylim_non_detect_band.filt_name 
                if first_Lylim_non_detect_band is not None else None
        }
        # if no bands bluewards of Lyman alpha,
        # select the galaxy by default
        if first_Lylim_non_detect_band is None:
            kwargs_out["SNR"] = None
            kwargs_out["mask"] = None
            return False, kwargs_out
        # crop gal to not include any ignore_bands
        gal_filt_names = np.array(
            [
                filt_name for filt_name in 
                gal.aper_phot[self.aper_diam].filterset.filt_names
                if filt_name not in ignore_bands
            ]
        )
        gal_SNRs = np.array(
            [
                SNR for filt_name, SNR in 
                zip(
                    gal.aper_phot[self.aper_diam].filterset.filt_names,
                    gal.aper_phot[self.aper_diam].SNR
                ) if filt_name not in ignore_bands
            ]
        )
        # find index of first Lya non-detect band
        first_Lylim_non_detect_index = np.where(
            gal_filt_names == first_Lylim_non_detect_band
        )[0][0]
        SNR_non_detect = gal_SNRs[: first_Lylim_non_detect_index + 1]
        kwargs_out["SNR"] = ",".join(SNR_non_detect.astype(str))
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            mask_non_detect = np.full(len(SNR_non_detect), False)
        else:
            gal_flux_mask = np.array(
                [
                    mask for filt_name, mask in
                    zip(
                        gal.aper_phot[self.aper_diam].filterset.filt_names,
                        gal.aper_phot[self.aper_diam].flux.mask
                    ) if filt_name not in ignore_bands
                ]
            )
            mask_non_detect = gal_flux_mask[: first_Lylim_non_detect_index + 1]
        kwargs_out["mask"] = ",".join(mask_non_detect.astype(str))
        # require the first Lylim non detect band and all bluewards bands 
        # to be non-detected at < SNR_lim if not masked
        selected = all(
            SNR < self.kwargs["SNR_lim"] or mask 
            for mask, SNR in zip(mask_non_detect, SNR_non_detect)
        )
        return selected, kwargs_out


class Bluewards_Lya_Non_Detect_Selector(Redshift_Selector):
    """Select galaxies non-detected in all bands bluewards of Lyman-alpha.

    Requires the first band entirely bluewards of the (redshifted)
    Lyman-alpha line, and all bands further bluewards, to have
    `SNR < SNR_lim` or be masked. Galaxies with no bands bluewards of
    Lyman-alpha are not selected.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry/SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run providing the best-fit redshift used to
        place the Lyman-alpha line.
    SNR_lim : `float`
        Signal-to-noise ratio non-detection threshold.
    fit_filterset : `Multiple_Filter`, optional
        If given, restrict the bands considered to those in this
        filterset. Default is `None`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        SNR_lim: float,
        fit_filterset: Optional[Multiple_Filter] = None,
    ):
        kwargs = {"SNR_lim": SNR_lim}
        super().__init__(
            aper_diam,
            SED_fitter,
            fit_filterset = fit_filterset,
            **kwargs
        )

    @property
    def _selection_name(self) -> str:
        selection_name = f"blue_Lya<{self.kwargs['SNR_lim']:.1f}"
        # if self.kwargs["ignore_bands"] is not None:
        #     ignore_str = ",".join(self.kwargs["ignore_bands"])
        #     selection_name += f"_no_{ignore_str}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim"]#, "ignore_bands"]

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic non-detection kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            Names `["1st_filt", "SNR", "mask"]` (the first Lyman-alpha
            non-detect band, comma-separated SNRs, and comma-separated
            mask flags of the bands checked) and their `str` dtypes.
        """
        return ["1st_filt", "SNR", "mask"], [str, str, str]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            # if self.kwargs["ignore_bands"] is not None:
            #     for band in self.kwargs["ignore_bands"]:
            #         # ensure this band exists
            #         assertions.extend([band in json.loads(config.get("Other", "ALL_BANDS"))])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        # extract first Lya non-detect band
        from .Emission_lines import line_diagnostics

        if self.fit_filterset is None:
            ignore_bands = []
        else:
            ignore_bands = [
                filt_name for filt_name in 
                gal.aper_phot[self.aper_diam].filterset.filt_names
                if filt_name not in self.fit_filterset.filt_names
            ]
        first_Lya_non_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fitter.label].phot_rest. \
                get_first_bluewards_band(
                    line_diagnostics["Lya"]["line_wav"],
                    ignore_bands,
                )
        kwargs_out = {"1st_filt": first_Lya_non_detect_band}
        # if no bands bluewards of Lyman alpha, 
        # select the galaxy by default
        if first_Lya_non_detect_band is None:
            kwargs_out["SNR"] = None
            kwargs_out["mask"] = None
            return False, kwargs_out
        
        gal_filt_names = np.array(
            [
                filt_name for filt_name in 
                gal.aper_phot[self.aper_diam].filterset.filt_names
                if filt_name not in ignore_bands
            ]
        )
        gal_SNRs = np.array(
            [
                SNR for filt_name, SNR in 
                zip(
                    gal.aper_phot[self.aper_diam].filterset.filt_names,
                    gal.aper_phot[self.aper_diam].SNR
                ) if filt_name not in ignore_bands
            ]
        )
        # find index of first Lya non-detect band
        first_Lya_non_detect_index = np.where(
            gal_filt_names == first_Lya_non_detect_band
        )[0][0]
        SNR_non_detect = gal_SNRs[: first_Lya_non_detect_index + 1]
        kwargs_out["SNR"] = ",".join(np.array(SNR_non_detect).astype(str))
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            mask_non_detect = np.full(len(SNR_non_detect), False)
        else:
            gal_flux_mask = np.array(
                [
                    mask for filt_name, mask in
                    zip(
                        gal.aper_phot[self.aper_diam].filterset.filt_names,
                        gal.aper_phot[self.aper_diam].flux.mask
                    ) if filt_name not in ignore_bands
                ]
            )
            mask_non_detect = gal_flux_mask[: first_Lya_non_detect_index + 1]
        kwargs_out["mask"] = ",".join(np.array(mask_non_detect).astype(str))
        # require the first Lya non detect band and all bluewards bands 
        # to be non-detected at < SNR_lim if not masked
        selected = all(
            SNR < self.kwargs["SNR_lim"] or mask 
            for mask, SNR in zip(mask_non_detect, SNR_non_detect)
        )
        return selected, kwargs_out


class Redwards_Lya_Detect_Selector(Redshift_Selector):
    """Select galaxies detected in bands redwards of Lyman-alpha.

    Requires the first band redwards of the (redshifted) Lyman-alpha
    line, and all bands further redwards (optionally restricted to
    wide bands), to have `SNR > SNR_lims` (a single limit applied to
    all bands, or the n^th such band compared to `SNR_lims[n]`) or be
    masked. Galaxies with no bands redwards of Lyman-alpha are not
    selected.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry/SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run providing the best-fit redshift used to
        place the Lyman-alpha line.
    SNR_lims : `float`
        Signal-to-noise ratio detection threshold(s): either a single
        value applied to every redwards band, or a list/array giving
        the threshold for the n^th redwards band.
    widebands_only : `bool`
        If `True`, only consider wide (``"W"``/``"LP"``) bands
        redwards of Lyman-alpha.
    fit_filterset : `Multiple_Filter`, optional
        If given, restrict the bands considered to those in this
        filterset. Default is `None`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        SNR_lims: float,
        widebands_only: bool,
        fit_filterset: Optional[Multiple_Filter] = None,
    ):
        kwargs = {
            "SNR_lims": SNR_lims,
            "widebands_only": widebands_only,
        }
        super().__init__(
            aper_diam,
            SED_fitter,
            fit_filterset = fit_filterset,
            **kwargs
        )

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
        # if self.kwargs["ignore_bands"] is not None:
        #     ignore_str = ",".join(self.kwargs["ignore_bands"])
        #     selection_name += f"_no_{ignore_str}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lims", "widebands_only"]#, "ignore_bands"]

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic detection kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            Names `["1st_filt", "bands", "SNR", "mask"]` (the first
            redwards-of-Lya band, the comma-separated bands checked,
            their comma-separated SNRs, and comma-separated mask
            flags) and their `str` dtypes.
        """
        return ["1st_filt", "bands", "SNR", "mask"], [str, str, str, str]

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
            # if self.kwargs["ignore_bands"] is not None:
            #     for band in self.kwargs["ignore_bands"]:
            #         # ensure this band exists
            #         assertions.extend([band in json.loads(config.get("Other", "ALL_BANDS"))])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        from .Emission_lines import line_diagnostics
        if self.fit_filterset is None:
            ignore_bands = []
        else:
            ignore_bands = [
                filt_name for filt_name in 
                gal.aper_phot[self.aper_diam].filterset.filt_names
                if filt_name not in self.fit_filterset.filt_names
            ]
        gal_filt_names = np.array(
            [
                filt_name for filt_name in 
                gal.aper_phot[self.aper_diam].filterset.filt_names
                if filt_name not in ignore_bands
            ]
        )
        if isinstance(self.kwargs["SNR_lims"], (int, float)):
            SNR_lims = np.full(len(gal_filt_names), self.kwargs["SNR_lims"])
        else:
            SNR_lims = self.kwargs["SNR_lims"]

        # extract first Lya non-detect band
        first_Lya_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fitter.label].phot_rest. \
                get_first_redwards_band(
                    line_diagnostics["Lya"]["line_wav"],
                    ignore_bands = ignore_bands,
                )
        kwargs_out = {"1st_filt": first_Lya_detect_band}
        # if no bands redwards of Lyman alpha, do not select the galaxy
        if first_Lya_detect_band is None:
            kwargs_out["bands"] = None
            kwargs_out["SNR"] = None
            kwargs_out["mask"] = None
            return False, kwargs_out
        # find index of first Lya non-detect band
        first_Lya_detect_index = np.where(
            gal_filt_names == first_Lya_detect_band
        )[0][0]
        detect_bands = gal_filt_names[first_Lya_detect_index:]
        gal_SNRs = np.array(
            [
                SNR for filt_name, SNR in 
                zip(
                    gal.aper_phot[self.aper_diam].filterset.filt_names,
                    gal.aper_phot[self.aper_diam].SNR
                ) if filt_name not in ignore_bands
            ]
        )
        SNR_detect = gal_SNRs[first_Lya_detect_index:]
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            mask_detect = np.full(len(SNR_detect), False)
        else:
            gal_flux_mask = np.array(
                [
                    mask for filt_name, mask in
                    zip(
                        gal.aper_phot[self.aper_diam].filterset.filt_names,
                        gal.aper_phot[self.aper_diam].flux.mask
                    ) if filt_name not in ignore_bands
                ]
            )
            mask_detect = gal_flux_mask[first_Lya_detect_index:]
        # option as to whether to exclude potentially 
        # shallower medium/narrow bands in this calculation
        if self.kwargs["widebands_only"]:
            wide_band_detect_indices = [
                True if "W" in band.upper() or "LP" in band.upper()
                else False for band in detect_bands
            ]
            kwargs_out["bands"] = ",".join(np.array(detect_bands)[wide_band_detect_indices].astype(str))
            SNR_detect = np.array(SNR_detect)[wide_band_detect_indices]
            mask_detect = np.array(mask_detect)[wide_band_detect_indices]
        else:
            kwargs_out["bands"] = ",".join(np.array(detect_bands).astype(str))
        
        kwargs_out["SNR"] = ",".join(SNR_detect.astype(str))
        kwargs_out["mask"] = ",".join(mask_detect.astype(str))
        # selection criteria
        selected = all(
            SNR > SNR_lim or mask for mask, SNR, SNR_lim
            in zip(mask_detect, SNR_detect, SNR_lims)
        )
        return selected, kwargs_out


class Lya_Band_Selector(Redshift_Selector):
    """Select galaxies by SNR in the band(s) straddling Lyman-alpha.

    Considers the bands between the first band redwards and the first
    band bluewards of the (redshifted) Lyman-alpha line, and requires
    either all of them to be detected or all of them to be
    non-detected at `SNR_lim`, depending on `detect_or_non_detect`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry/SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run providing the best-fit redshift used to
        place the Lyman-alpha line.
    SNR_lim : `int` or `float`
        Signal-to-noise ratio threshold.
    detect_or_non_detect : `str`
        Either ``"detect"`` to require `SNR > SNR_lim` in every band
        straddling Lyman-alpha, or ``"non_detect"`` to require
        `SNR < SNR_lim`.
    widebands_only : `bool`
        If `True`, only consider wide (``"W"``/``"LP"``) bands.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        SNR_lim: Union[int, float],
        detect_or_non_detect: str,
        widebands_only: bool,
    ):
        kwargs = {
            "SNR_lim": SNR_lim,
            "detect_or_non_detect": detect_or_non_detect,
            "widebands_only": widebands_only,
        }
        super().__init__(aper_diam, SED_fitter, **kwargs)

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

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic detection kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            Names `["1st_Lya_detect", "1st_Lya_nondetect",
            "detect_bands", "SNR"]` and their `str` dtypes.
        """
        kwarg_names = [
            "1st_Lya_detect",
            "1st_Lya_nondetect",
            "detect_bands",
            "SNR",
        ]
        kwarg_dtypes = [str, str, str, str]
        return kwarg_names, kwarg_dtypes

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
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        from .Emission_lines import line_diagnostics
        bands = np.array(gal.aper_phot[self.aper_diam].filterset.filt_names)
        # determine Lya band(s) - usually a single band, 
        # but could be two in the case of medium bands
        first_Lya_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fitter.label].phot_rest. \
            get_first_redwards_band(
                line_diagnostics["Lya"]["line_wav"],
                None, #self.kwargs["ignore_bands"],
            )
        first_Lya_non_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fitter.label].phot_rest. \
            get_first_bluewards_band(
                line_diagnostics["Lya"]["line_wav"],
                None, #self.kwargs["ignore_bands"],
            )
        if first_Lya_detect_band is None or first_Lya_non_detect_band is None:
            kwargs_out = {
                "1st_Lya_detect": first_Lya_detect_band,
                "1st_Lya_nondetect": first_Lya_non_detect_band,
                "detect_bands": None,
                "SNR": None,
                #"mask": None,
            }
            return False, kwargs_out
        first_Lya_detect_index = np.where(bands == \
            first_Lya_detect_band)[0][0]
        first_Lya_non_detect_index = np.where(bands == \
            first_Lya_non_detect_band)[0][0]
        # load SNRs, cropping by the relevant bands
        detect_bands = bands[
            first_Lya_detect_index : first_Lya_non_detect_index + 1
        ]
        kwargs_out = {
            "1st_Lya_detect": first_Lya_detect_band,
            "1st_Lya_nondetect": first_Lya_non_detect_band,
            "detect_bands": ",".join(np.array([band for band in detect_bands]).astype(str)),
        }
        if len(detect_bands) == 0:
            kwargs_out["SNR"] = None
            #kwargs_out["mask"] = None
            return False, kwargs_out
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
        kwargs_out["SNR"] = ",".join(SNRs.astype(str))
        #kwargs_out["mask"] = ",".join(mask_bands.astype(str))
        if len(SNRs) == 0:
            return False, kwargs_out
        if self.kwargs["detect_or_non_detect"].lower() == "detect":
            selected = all(
                SNR > self.kwargs["SNR_lim"] for SNR in SNRs
            )
        else:
            selected = all(
                SNR < self.kwargs["SNR_lim"] for SNR in SNRs
            )
        return selected, kwargs_out


class Unmasked_Bluewards_Lya_Selector(Redshift_Selector, Mask_Selector):
    """Select galaxies unmasked in the bands closest to Lyman-alpha (bluewards).

    Requires the last `n_bands` bands bluewards of the (redshifted)
    Lyman-alpha line (i.e. those closest to the break) to be unmasked.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry/SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run providing the best-fit redshift used to
        place the Lyman-alpha line.
    n_bands : `int`, optional
        Number of bands closest to Lyman-alpha (bluewards side)
        required to be unmasked. Default is `None`.
    widebands_only : `bool`, optional
        If `True`, only consider wide (``"W"``/``"LP"``) bands
        bluewards of Lyman-alpha. Default is `False`.
    fit_filterset : `Multiple_Filter`, optional
        If given, restrict the bands considered to those in this
        filterset. Default is `None`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        n_bands: Optional[int] = None,
        widebands_only: bool = False,
        fit_filterset: Optional[Multiple_Filter] = None,
    ):
        kwargs = {
            "n_bands": n_bands,
            "widebands_only": widebands_only,
        }
        # # this calls self._assertions with self.aper_diam = None and self.SED_fitter = None
        # Mask_Selector.__init__(self, **kwargs)
        Redshift_Selector.__init__(
            self,
            aper_diam,
            SED_fitter,
            fit_filterset = fit_filterset,
            **kwargs
        )

    @property
    def _selection_name(self) -> str:
        selection_name = "unmask"
        if self.kwargs["n_bands"] is not None:
            selection_name += f"_{self.kwargs['n_bands']}"
        if self.kwargs["widebands_only"]:
            selection_name += "_wide"
        # if self.kwargs["ignore_bands"] is not None:
        #     ignore_str = ",".join(self.kwargs["ignore_bands"])
        #     selection_name += f"_no_{ignore_str}"
        selection_name += "_blue_Lya"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return [
            "n_bands",
            "widebands_only",
            # "ignore_bands",
        ]
    
    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic unmasking kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["1st_filt", "n_unmask"]` and `[str, int]`.
        """
        return ["1st_filt", "n_unmask"], [str, int]

    def _assertions(self: Self) -> bool:
        try:
            assert isinstance(self.kwargs["n_bands"], int)
            assert self.kwargs["n_bands"] > 0
            assert isinstance(self.kwargs["widebands_only"], bool)
            # if self.kwargs["ignore_bands"] is not None:
            #     for band in self.kwargs["ignore_bands"]:
            #         # ensure this band exists
            #         assert band in json.loads(config.get("Other", "ALL_BANDS"))
            passed = True
        except:
            passed = False
        return passed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        return False

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        non_detect_filt_names, first_filt = self.get_non_detect_filt_names(
            gal.cat_filterset,
            gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].z
        )
        kwargs_out = {"1st_filt": first_filt}
        if first_filt is None:
            kwargs_out["n_unmask"] = 0
            return False, kwargs_out

        n_unmask = 0
        unmasked = np.full(len(non_detect_filt_names), False)
        for i, filt_name in enumerate(non_detect_filt_names):
            gal_filt_names = np.array(gal.aper_phot[self.aper_diam].filterset.filt_names)
            if filt_name in gal_filt_names:
                if isinstance(gal.aper_phot[self.aper_diam].flux, Masked):
                    gal_filt_index = np.where(gal_filt_names == filt_name)[0][0]
                    unmasked_filt = not gal.aper_phot[self.aper_diam].flux.mask[gal_filt_index]
                else:
                    unmasked_filt = True
                unmasked[i] = unmasked_filt
                if unmasked_filt:
                    n_unmask += 1
        kwargs_out["n_unmask"] = n_unmask

        if isinstance(self.kwargs["n_bands"], str):
            if self.kwargs["n_bands"].lower() == "any":
                selected = any(unmasked)
            else: # self.kwargs["n_bands"].lower() == "all"
                selected = all(unmasked)
        else: # isinstance(self.kwargs["n_bands"], int):
            selected = all(unmasked[-self.kwargs["n_bands"]:])
        return selected, kwargs_out
    
    def get_non_detect_filt_names(self: Self, filterset: Multiple_Filter, z: float) -> Optional[List[str]]:
        """Determine the band names bluewards of Lyman-alpha at a given redshift.

        Parameters
        ----------
        filterset : `Multiple_Filter`
            The filterset to search for bands bluewards of the
            (redshifted) Lyman-alpha line.
        z : `float`
            Redshift used to place the Lyman-alpha line.

        Returns
        -------
        `tuple` of (`numpy.ndarray` of `str` or `None`, `str` or `None`)
            The band names bluewards of (and including) the first band
            bluewards of Lyman-alpha (restricted to widebands if
            `widebands_only`), and the name of that first band. Both
            are `None` if there are no bands bluewards of Lyman-alpha.
        """
        from .Emission_lines import line_diagnostics
        if self.fit_filterset is None:
            ignore_bands = []
        else:
            ignore_bands = [
                filt_name for filt_name in filterset.filt_names
                if filt_name not in self.fit_filterset.filt_names
            ]
        first_Lya_non_detect_band = funcs.get_first_bluewards_band(
            z,
            filterset,
            line_diagnostics["Lya"]["line_wav"],
            ignore_bands = ignore_bands,
        )
        if first_Lya_non_detect_band is None:
            return None, first_Lya_non_detect_band
        
        # determine bands that require unmasking
        fit_filt_names = np.array(
            [
                filt_name for filt_name in filterset.filt_names
                if filt_name not in ignore_bands
            ]
        )
        # get index of first Lya non-detect band
        try:
            first_Lya_non_detect_index = np.where(
                fit_filt_names == first_Lya_non_detect_band
            )[0][0]
        except Exception as e:
            print(f"Error finding index of first Lya non-detect band: {e}")
            breakpoint()
            raise Exception()
        non_detect_filt_names = fit_filt_names[: first_Lya_non_detect_index + 1]
        if self.kwargs["widebands_only"]:
            wide_band_detect_indices = [
                True
                if "W" in filt_name.upper() or "LP" in filt_name.upper()
                else False
                for filt_name in non_detect_filt_names
            ]
            non_detect_filt_names = non_detect_filt_names[wide_band_detect_indices]
        return non_detect_filt_names, first_Lya_non_detect_band

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load the combined mask over bands bluewards of Lyman-alpha.

        Parameters
        ----------
        data : `Data`
            The dataset to construct the mask from.
        invert : `bool`, optional
            If `True`, invert the combined mask before returning it.
            Default is `True`.
        **kwargs
            Must include `"z"` (`float`), the redshift used to select
            the appropriate redshift bin (via `extract_zbins`) and
            determine the relevant bluewards bands.

        Returns
        -------
        `astropy.units.Quantity` or array-like
            The (optionally inverted) logical AND of the last
            `n_bands` bluewards-of-Lya band masks.

        Raises
        ------
        AssertionError
            If `"z"` is missing from `kwargs`, is not a `float`, does
            not fall within exactly one redshift bin from
            `extract_zbins`, or if the individual band masks do not
            all have the same shape.
        """
        #try:
        assert "z" in kwargs.keys(), \
            galfind_logger.critical(
                "Redshift must be provided to load the mask."
            )
        z = kwargs["z"]
        assert isinstance(z, float), \
            galfind_logger.critical(
                f"'z' must be a float, not {type(z)}."
            )
        zbins = self.extract_zbins(data)
        # choose zbin which contains the redshift
        zbin = [zbin for zbin in zbins if zbin[0] <= z < zbin[1]]
        assert len(zbin) == 1, \
            galfind_logger.critical(
                f"Redshift {z} not in any zbin: {zbins}."
            )
        zbin = zbin[0]
        # determine bands bluewards of Lya
        non_detect_filt_names = self.get_non_detect_filt_names(data.filterset, z)[0]

        # load mask from each redwards band
        masks = np.array([
            data[filt_name].load_mask("MASK")[0].astype(bool)
            for filt_name in non_detect_filt_names
        ])
        assert all(mask.shape == masks[0].shape for mask in masks), \
        galfind_logger.critical(
            f"Mask shapes do not match: {[mask.shape for mask in masks]}"
        )
        if isinstance(self.kwargs["n_bands"], str):
            if self.kwargs["n_bands"].lower() == "any":
                combined_mask = np.logical_or.reduce(masks)
            else: # self.kwargs["n_bands"].lower() == "all"
                combined_mask = np.logical_and.reduce(masks)
        else:
            combined_mask = np.logical_and.reduce(masks[-self.kwargs["n_bands"] :])

        if not invert:
            combined_mask = np.logical_not(combined_mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        # except Exception as e:
        #     galfind_logger.error(
        #         f"Error loading {repr(self)} mask: {e}"
        #     )
        #     breakpoint()
        #     raise e
        return combined_mask

    def extract_zbins(
        self: Self,
        data: Data,
    ) -> List[float]:
        """Determine redshift bins over which the bluewards band set is fixed.

        Parameters
        ----------
        data : `Data`
            The dataset whose filterset determines the redshift bin
            edges, computed from each band's 50%-transmission upper
            wavelength relative to Lyman-alpha.

        Returns
        -------
        `list` of `list` of `float`
            Consecutive `[z_low, z_high]` pairs, one fewer than the
            number of bands, over which the set of bands bluewards of
            Lyman-alpha does not change.
        """
        # determine redshift limits at which bands become bluewards of Lya
        from .Emission_lines import line_diagnostics
        if self.fit_filterset is None:
            ignore_bands = []
        else:
            ignore_bands = [
                filt_name for filt_name in data.filterset.filt_names
                if filt_name not in self.fit_filterset.filt_names
            ]
        zlims = [
            ((filt.WavelengthUpper50 / line_diagnostics["Lya"]["line_wav"]) - 1.0).value
            for filt in data.filterset if filt.filt_name not in ignore_bands
        ]
        zbins = [[a, b] for a, b in zip(zlims, zlims[1:])]
        return zbins


class Unmasked_Redwards_Lya_Selector(Redshift_Selector, Mask_Selector):
    """Select galaxies unmasked in the bands closest to Lyman-alpha (redwards).

    Requires the first `n_bands` bands redwards of the (redshifted)
    Lyman-alpha line (i.e. those closest to the break), or all/any of
    them, to be unmasked.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry/SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run providing the best-fit redshift used to
        place the Lyman-alpha line.
    n_bands : `int` or `str`
        Number of bands closest to Lyman-alpha (redwards side)
        required to be unmasked, or ``"all"``/``"any"`` to require all
        or any redwards bands to be unmasked.
    widebands_only : `bool`, optional
        If `True`, only consider wide (``"W"``/``"LP"``) bands
        redwards of Lyman-alpha. Default is `False`.
    fit_filterset : `Multiple_Filter`, optional
        If given, restrict the bands considered to those in this
        filterset. Default is `None`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        n_bands: Union[int, str],
        widebands_only: bool = False,
        fit_filterset: Optional[Multiple_Filter] = None,
    ):
        kwargs = {"n_bands": n_bands, "widebands_only": widebands_only}
        # # this calls self._assertions with self.aper_diam = None and self.SED_fitter = None
        # Mask_Selector.__init__(self, **kwargs)
        Redshift_Selector.__init__(self, aper_diam, SED_fitter, fit_filterset = fit_filterset,**kwargs)

    @property
    def _selection_name(self) -> str:
        if isinstance(self.kwargs["n_bands"], str):
            # require all redwards bands to be detected at >SNR_lims
            selection_name = "unmask_ALL_red_Lya"
        else:
            selection_name = f"unmask_{self.kwargs['n_bands']}_red_Lya"
        if self.kwargs["widebands_only"]:
            selection_name += "_wide"
        # if self.kwargs["ignore_bands"] is not None:
        #     ignore_str = ",".join(self.kwargs["ignore_bands"])
        #     selection_name += f"_no_{ignore_str}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["n_bands", "widebands_only"] #, "ignore_bands"]

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic unmasking kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["1st_filt", "n_unmask"]` and `[str, int]`.
        """
        return ["1st_filt", "n_unmask"], [str, int]

    def _assertions(self: Self) -> bool:
        try:
            assert isinstance(self.kwargs["n_bands"], (int, str))
            if isinstance(self.kwargs["n_bands"], int):
                assert self.kwargs["n_bands"] > 0, \
                    galfind_logger.critical(
                        f"{self.kwargs['n_bands']} must be a positive integer."
                    )
            else: # isinstance(self.kwargs["n_bands"], str):
                assert self.kwargs["n_bands"].lower() in ["all", "any"], \
                    galfind_logger.critical(
                        f"{self.kwargs['n_bands']} must be 'all' or 'any'."
                    )
            assert isinstance(self.kwargs["widebands_only"], bool)
            # if self.kwargs["ignore_bands"] is not None:
            #     for band in self.kwargs["ignore_bands"]:
            #         # ensure this band exists
            #         assert band in json.loads(config.get("Other", "ALL_BANDS"))
            passed = True
        except:
            passed = False
        return passed
        
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        
        detect_filt_names, first_filt = self.get_detect_filt_names(
            gal.cat_filterset,
            gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].z
        )
        kwargs_out = {"1st_filt": first_filt}
        if first_filt is None:
            kwargs_out["n_unmask"] = 0
            return False, kwargs_out

        n_unmask = 0
        unmasked = np.full(len(detect_filt_names), False)
        for i, filt_name in enumerate(detect_filt_names):
            gal_filt_names = np.array(gal.aper_phot[self.aper_diam].filterset.filt_names)
            if filt_name in gal_filt_names:
                if isinstance(gal.aper_phot[self.aper_diam].flux, Masked):
                    gal_filt_index = np.where(gal_filt_names == filt_name)[0][0]
                    unmasked_filt = not gal.aper_phot[self.aper_diam].flux.mask[gal_filt_index]
                else:
                    unmasked_filt = True
                unmasked[i] = unmasked_filt
                if unmasked_filt:
                    n_unmask += 1
        kwargs_out["n_unmask"] = n_unmask

        if isinstance(self.kwargs["n_bands"], str):
            if self.kwargs["n_bands"].lower() == "any":
                selected = any(unmasked)
            else: # self.kwargs["n_bands"].lower() == "all"
                selected = all(unmasked)
        else: # isinstance(self.kwargs["n_bands"], int):
            selected = all(unmasked[:self.kwargs["n_bands"]])
        return selected, kwargs_out

    def get_detect_filt_names(
        self: Self,
        filterset: Multiple_Filter,
        z: float,
    ) -> Optional[List[str]]:
        """Determine the band names redwards of Lyman-alpha at a given redshift.

        Parameters
        ----------
        filterset : `Multiple_Filter`
            The filterset to search for bands redwards of the
            (redshifted) Lyman-alpha line.
        z : `float`
            Redshift used to place the Lyman-alpha line.

        Returns
        -------
        `tuple` of (`numpy.ndarray` of `str` or `None`, `str` or `None`)
            The band names from the first band redwards of Lyman-alpha
            onwards (restricted to widebands if `widebands_only`), and
            the name of that first band. Both are `None` if there are
            no bands redwards of Lyman-alpha.
        """
        from .Emission_lines import line_diagnostics
        if self.fit_filterset is None:
            ignore_bands = []
        else:
            ignore_bands = [
                filt_name for filt_name in filterset.filt_names
                if filt_name not in self.fit_filterset.filt_names
            ]
        first_Lya_detect_band = funcs.get_first_redwards_band(
            z,
            filterset,
            line_diagnostics["Lya"]["line_wav"],
            ignore_bands = ignore_bands
        )
        if first_Lya_detect_band is None:
            return None, first_Lya_detect_band
        
        # determine bands that require unmasking
        fit_filt_names = np.array(
            [
                filt_name for filt_name in filterset.filt_names
                if filt_name not in ignore_bands
            ]
        )
        # get index of first Lya detect band
        first_Lya_detect_index = np.where(
            fit_filt_names == first_Lya_detect_band
        )[0][0]
        detect_filt_names = fit_filt_names[first_Lya_detect_index:] #[: first_Lya_non_detect_index + 1]
        if self.kwargs["widebands_only"]:
            wide_band_detect_indices = [
                True
                if "W" in filt_name.upper() or "LP" in filt_name.upper()
                else False
                for filt_name in detect_filt_names
            ]
            detect_filt_names = detect_filt_names[wide_band_detect_indices]
        return detect_filt_names, first_Lya_detect_band

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Load the combined mask over bands redwards of Lyman-alpha.

        Parameters
        ----------
        data : `Data`
            The dataset to construct the mask from.
        invert : `bool`, optional
            If `True`, invert the combined mask before returning it.
            Default is `True`.
        **kwargs
            Must include `"z"` (`float`), the redshift used to select
            the appropriate redshift bin (via `extract_zbins`) and
            determine the relevant redwards bands.

        Returns
        -------
        `astropy.units.Quantity` or `None`
            The (optionally inverted) combined mask over the first
            `n_bands` redwards-of-Lya bands, or `None` if an error
            occurs while loading it (also re-raised as an exception).

        Raises
        ------
        AssertionError
            If `"z"` is missing from `kwargs`, is not a `float`, does
            not fall within exactly one redshift bin from
            `extract_zbins`, or if the individual band masks do not
            all have the same shape.
        """
        try:
            assert "z" in kwargs.keys(), \
                galfind_logger.critical(
                    "Redshift must be provided to load the mask."
                )
            z = kwargs["z"]
            assert isinstance(z, float), \
                galfind_logger.critical(
                    f"'z' must be a float, not {type(z)}."
                )
            zbins = self.extract_zbins(data)
            # choose zbin which contains the redshift
            zbin = [zbin for zbin in zbins if zbin[0] <= z < zbin[1]]
            assert len(zbin) == 1, \
                galfind_logger.critical(
                    f"Redshift {z} in zbins: {zbin}."
                )
            #zbin = zbin[0]
            # determine bands bluewards of Lya
            detect_filt_names = self.get_detect_filt_names(data.filterset, z)[0]

            # load mask from each redwards band
            masks = np.array([
                data[filt_name].load_mask("MASK")[0].astype(bool)
                for filt_name in detect_filt_names
            ])
            assert all(mask.shape == masks[0].shape for mask in masks), \
            galfind_logger.critical(
                f"Mask shapes do not match: {[mask.shape for mask in masks]}"
            )
            if isinstance(self.kwargs["n_bands"], str):
                if self.kwargs["n_bands"].lower() == "any":
                    combined_mask = np.logical_or.reduce(masks)
                else: # self.kwargs["n_bands"].lower() == "all"
                    combined_mask = np.logical_and.reduce(masks)
            else:
                combined_mask = np.logical_and.reduce(masks[: self.kwargs["n_bands"]])
            if not invert:
                combined_mask = np.logical_not(combined_mask)
            galfind_logger.info(
                f"Loaded {repr(self)} mask from {data.survey}!"
            )
        except Exception as e:
            galfind_logger.critical(
                f"Error loading {repr(self)} mask: {e}"
            )
            combined_mask = None
            breakpoint()
            raise e
        return combined_mask
    
    def extract_zbins(
        self: Self,
        data: Data,
    ) -> List[float]:
        """Determine redshift bins over which the redwards band set is fixed.

        Parameters
        ----------
        data : `Data`
            The dataset whose filterset determines the redshift bin
            edges, computed from each band's 50%-transmission lower
            wavelength relative to Lyman-alpha.

        Returns
        -------
        `list` of `list` of `float`
            Consecutive `[z_low, z_high]` pairs, one fewer than the
            number of bands, over which the set of bands redwards of
            Lyman-alpha does not change.
        """
        # determine redshift limits at which bands become bluewards of Lya
        from .Emission_lines import line_diagnostics
        if self.fit_filterset is None:
            ignore_bands = []
        else:
            ignore_bands = [
                filt_name for filt_name in data.filterset.filt_names
                if filt_name not in self.fit_filterset.filt_names
            ]
        zlims = [
            ((filt.WavelengthLower50 / line_diagnostics["Lya"]["line_wav"]) - 1.0).value
            for filt in data.filterset if filt.filt_name not in ignore_bands
        ]
        zbins = [[a, b] for a, b in zip(zlims, zlims[1:])]
        return zbins


class Band_Mag_Selector(Photometry_Selector):
    """Select galaxies by AB magnitude detection/non-detection in one band.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry was extracted at.
    band : `str` or `int`
        The band to select on, either by name or by index into the
        galaxy's filterset (e.g. `0` for the bluest band, `-1` for the
        reddest).
    detect_or_non_detect : `str`
        Either ``"detect"`` to select `mag < mag_lim`, or
        ``"non_detect"`` to select `mag > mag_lim`. A masked band never
        selects, regardless of this setting.
    mag_lim : `int` or `float`
        AB magnitude limit to compare against.
    """

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
    def _selection_name(self: Self) -> str:
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
    def _include_kwargs(self: Self) -> List[str]:
        return ["band", "mag_lim", "detect_or_non_detect"]

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic `mag`/`masked` kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["mag", "masked"]` and `[float, bool]`.
        """
        return ["mag", "masked"], [float, bool]

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
        **kwargs,
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            try:
                failed = self.kwargs["band"] not in \
                    gal.aper_phot[self.aper_diam].filterset.filt_names
            except:
                failed = True
            return failed
        else:
            return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        if isinstance(self.kwargs["band"], str):
            band_index = int(np.where(np.array( \
                gal.aper_phot[self.aper_diam].filterset.filt_names) \
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
        selected = (
            not masked
            and ((self.kwargs["detect_or_non_detect"].lower() \
            == "detect" and mag < self.kwargs["mag_lim"])
            or (self.kwargs["detect_or_non_detect"].lower() \
            == "non_detect" and mag > self.kwargs["mag_lim"]))
        )
        kwargs_out = {"mag": mag, "masked": masked}
        return selected, kwargs_out

    def _assert_cat(self: Self, cat: Catalogue) -> None:
        if isinstance(self.kwargs["band"], str):
            assert (self.kwargs["band"] in cat.filterset.filt_names), \
                galfind_logger.critical(
                    f"{self.kwargs['band']} not in {cat.filterset.filt_names}."
                )
        super()._assert_cat(cat)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Optional[Catalogue]:
        return Photometry_Selector._call_cat(self, cat, return_copy)


class Band_SNR_Selector(Photometry_Selector):
    """Select galaxies by signal-to-noise detection/non-detection in one band.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry was extracted at.
    band : `str` or `int`
        The band to select on, either by name or by index into the
        galaxy's filterset (e.g. `0` for the bluest band, `-1` for the
        reddest).
    detect_or_non_detect : `str`
        Either ``"detect"`` to select `SNR > SNR_lim`, or
        ``"non_detect"`` to select `SNR < SNR_lim`. A masked band never
        selects, regardless of this setting.
    SNR_lim : `int` or `float`
        Signal-to-noise ratio limit to compare against.
    """

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
        **kwargs,
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            try:
                failed = self.kwargs["band"] not in \
                    gal.aper_phot[self.aper_diam].filterset.filt_names
            except:
                failed = True
            return failed
        else:
            return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        if isinstance(self.kwargs["band"], str):
            band_index = int(np.where(np.array( \
                gal.aper_phot[self.aper_diam].filterset.filt_names) \
                == self.kwargs["band"])[0][0])
        else:
            band_index = self.kwargs["band"]
        SNR = gal.aper_phot[self.aper_diam].SNR[band_index]
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            masked = False
        else:
            masked = gal.aper_phot[self.aper_diam].flux.mask[band_index]
        # fails if masked
        selected = (
            not masked
            and ((self.kwargs["detect_or_non_detect"].lower() \
            == "detect" and SNR > self.kwargs["SNR_lim"])
            or (self.kwargs["detect_or_non_detect"].lower() \
            == "non_detect" and SNR < self.kwargs["SNR_lim"]))
        )
        kwargs_out = {"SNR": SNR, "masked": masked}
        return selected, kwargs_out

    def _assert_cat(self: Self, cat: Catalogue) -> None:
        if isinstance(self.kwargs["band"], str):
            assert (self.kwargs["band"] in cat.filterset.filt_names), \
                galfind_logger.critical(
                    f"{self.kwargs['band']} not in {cat.filterset.filt_names}."
                )
        super()._assert_cat(cat)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Catalogue]:
        return Photometry_Selector._call_cat(self, cat, return_copy, *args, **kwargs)


class Stacked_Blue_Lya_Non_Detect_Selector(SED_fit_Selector):
    """Select galaxies non-detected in the stacked bands bluewards of Lyman-alpha.

    Stacks (or uses singly, if only one band qualifies) all bands
    bluewards of the (redshifted, capped at `zmax`) Lyman-alpha line
    and requires the resulting SNR to be below `SNR_lim`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry/SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run providing the best-fit redshift used to
        place the Lyman-alpha line.
    SNR_lim : `int` or `float`
        Signal-to-noise ratio non-detection threshold for the stacked
        band(s).
    dz : `float`, optional
        Redshift buffer subtracted from the (capped) best-fit redshift
        when determining which bands are bluewards of Lyman-alpha.
        Default is `0.0`.
    zmax : `float`, optional
        Maximum redshift used in place of the best-fit redshift when
        it exceeds this value. Default is `15.0`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        #filterset: Union[str, List[str], Multiple_Filter],
        #detect_or_non_detect: str,
        SNR_lim: Union[int, float],
        dz: float = 0.0,
        zmax: float = 15.0,
    ):
        # # if string, turn filterset into a list
        # if isinstance(filterset, str):
        #     filterset = filterset.split("+")
        # # if list, turn filterset into a Multiple_Filter object
        # if isinstance(filterset, list):
        #     from . import Filter, Multiple_Filter
        #     all_bands = json.loads(config.get("Other", "ALL_BANDS"))
        #     assert all(band in all_bands for band in filterset), \
        #         galfind_logger.critical(
        #             f"One or more bands in {filterset} not in {all_bands}!"
        #         )
        #     filterset = Multiple_Filter(
        #         [Filter.from_filt_name(filt_name) for filt_name in filterset]
        #     )
        kwargs = {
            #"filterset": filterset,
            #"detect_or_non_detect": detect_or_non_detect,
            "SNR_lim": SNR_lim,
            "dz": dz,
            "zmax": zmax,
        }
        super().__init__(aper_diam, SED_fitter, **kwargs)

    @property
    def _selection_name(self) -> str:
        # if self.kwargs["detect_or_non_detect"].lower() == "detect":
        #     sign = ">"
        # else: # self.kwargs["detect_or_non_detect"].lower() == "non_detect"
        #     sign = "<"
        if self.kwargs["dz"] > 0.0:
            dz_str = f"_dz{self.kwargs['dz']:.2f}"
        else:
            dz_str = ""
        selection_name = f"blue_Lya_stacked_SNR<{self.kwargs['SNR_lim']:.1f}{dz_str}_zmax{self.kwargs['zmax']:.1f}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim", "dz", "zmax"] # "filterset", "detect_or_non_detect", 

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic stacked-band kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["bands", "SNR"]` and `[str, float]`.
        """
        return ["bands", "SNR"], [str, float]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            #assertions.extend([isinstance(self.kwargs["filterset"], Multiple_Filter)])
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            assertions.extend([isinstance(self.kwargs["dz"], (int, float))])
            assertions.extend([self.kwargs["dz"] >= 0.0])
            assertions.extend([isinstance(self.kwargs["zmax"], (int, float))])
            assertions.extend([self.kwargs["zmax"] > 0.0])
            #assertions.extend([self.kwargs["detect_or_non_detect"].lower() in ["detect", "non_detect"]])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        blue_stack_label = self._get_blue_stack_label(gal)
        if blue_stack_label is None:
            return True
        else:
            return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        blue_stack_label = self._get_blue_stack_label(gal)
        if not "+" in blue_stack_label:
            SNR = float(gal.aper_phot[self.aper_diam][blue_stack_label].SNR[0].unmasked)
        else:
            SNR = float(gal.aper_phot[self.aper_diam].stacked_band_snrs[blue_stack_label])
        kwargs_out = {"bands": blue_stack_label, "SNR": SNR}
        if not np.isfinite(SNR):
            selected = False
        else:
            selected = SNR < self.kwargs["SNR_lim"]
        return selected, kwargs_out

    def _assert_cat(
        self: Self,
        cat: Catalogue,
    ) -> None:
        req_stacked_bands = self._required_stacked_bands(cat.filterset)
        stacked_band_data_filt_names = [band_data.filt_name for band_data in cat.data.stacked_band_data_arr]
        if not all(["+".join(req_stacked_band.filt_names) in stacked_band_data_filt_names for req_stacked_band in req_stacked_bands]):
            err_message = f"{repr(self)} requires the following stacked bands: " + \
                f"{['+'.join(req_stacked_band.filt_names) for req_stacked_band in req_stacked_bands]}, " + \
                f"but only the following stacked bands are available: {stacked_band_data_filt_names}!"
            galfind_logger.critical(err_message)
            raise Exception(err_message)
        cat.load_stacked_band_data_snrs()
        super()._assert_cat(cat)

    def _required_stacked_bands(
        self: Self,
        filterset: Multiple_Filter,
    ) -> List[Multiple_Filter]:
        from . import Multiple_Filter, wav_lyman_alpha
        # determine required stacked bands from maximum redshift
        max_wav_lim = wav_lyman_alpha * u.AA * (1.0 + self.kwargs["zmax"] - self.kwargs["dz"])
        blue_filters = Multiple_Filter([filt for filt in filterset if filt.WavelengthUpper50 < max_wav_lim])
        req_stacked_bands = [
            Multiple_Filter(deepcopy(blue_filters)[:i+1]) for i in range(len(blue_filters))
        ][1:]
        return req_stacked_bands
    
    def _get_blue_stack_label(
        self: Self,
        gal: Galaxy,
    ) -> str:
        from . import Multiple_Filter, wav_lyman_alpha
        z = gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].z
        if z < 0.0:
            return None
        else:
            z = np.min((z, self.kwargs["zmax"]))
            blue_bands = [
                filt.filt_name for filt in gal.aper_phot[self.aper_diam].filterset \
                if filt.WavelengthUpper50 < wav_lyman_alpha * u.AA * (1.0 + z - self.kwargs["dz"])
            ]
            if len(blue_bands) == 0:
                return None
            elif len(blue_bands) == 1:
                return blue_bands[0]
            else:
                closest_blue_band_label = blue_bands[-1]
                label = [
                    key for key in gal.aper_phot[self.aper_diam].stacked_band_snrs.keys() \
                    if closest_blue_band_label in key
                ][0]
                assert label.split("+")[-1] == closest_blue_band_label, \
                    galfind_logger.critical(
                        f"{closest_blue_band_label=} not at end of {closest_blue_band_label}!"
                    )
                return label


# TODO: UVJ_Selector should be specific 
# case of a photometric property selector

# class UVJ_Selector(Selector):

#     @property
#     def _selection_name(self) -> str:
#         return f"UVJ_{self.kwargs['quiescent_or_star_forming'].lower()}"

#     @property
#     def _include_kwargs(self) -> List[str]:
#         return ["quiescent_or_star_forming"]

#     @property
#     def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
#         return ["U_minus_V", "V_minus_J"], [float, float]

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
#         *args,
#         **kwargs,
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
#         *args,
#         **kwargs,
#     ) -> Tuple[bool, Dict[str, Any]]:
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
#         selected = (self.kwargs["quiescent_or_star_forming"].lower() \
#             == "quiescent" and is_quiescent) or \
#             (self.kwargs["quiescent_or_star_forming"].lower() \
#             == "star_forming" and not is_quiescent)
#         kwargs_out = {"U_minus_V": U_minus_V, "V_minus_J": V_minus_J}
#         return selected, kwargs_out


class Chi_Sq_Lim_Selector(SED_fit_Selector):
    """Select galaxies with a (reduced) chi-squared below a limit.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run whose chi-squared this selector acts on.
    chi_sq_lim : `int` or `float`
        Chi-squared limit to compare against.
    reduced : `bool`
        If `True`, compare the reduced chi-squared (chi-squared
        divided by the number of unmasked bands minus one) to
        `chi_sq_lim`; otherwise compare the absolute chi-squared.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        chi_sq_lim: Union[int, float],
        reduced: bool,
    ):
        kwargs = {
            "chi_sq_lim": chi_sq_lim,
            "reduced": reduced,
        }
        super().__init__(aper_diam, SED_fitter, **kwargs)

    @property
    def _selection_name(self) -> str:
        selection_name = f"chi_sq<{self.kwargs['chi_sq_lim']:.1f}"
        if self.kwargs["reduced"]:
            selection_name = f"red_{selection_name}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_lim", "reduced"]
    
    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic chi-squared kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["chi_sq"]` and `[float]`, plus `["n_bands"]`/`[int]` if
            `reduced` is `True`.
        """
        kwarg_names = ["chi_sq"]
        kwarg_dtypes = [float]
        if self.kwargs["reduced"]:
            kwarg_names.append("n_bands")
            kwarg_dtypes.append(int)
        return kwarg_names, kwarg_dtypes

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["chi_sq_lim"], (int, float))])
            assertions.extend([self.kwargs["chi_sq_lim"] > 0.0])
            assertions.extend([isinstance(self.kwargs["reduced"], bool)])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        chi_sq_lim = self.kwargs["chi_sq_lim"]
        if self.kwargs["reduced"]:
            if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
                n_bands = len(gal.aper_phot[self.aper_diam].filterset.filt_names)
            else:
                n_bands = len(
                    [
                        mask_band
                        for mask_band in gal.aper_phot[self.aper_diam].flux.mask
                        if not mask_band
                    ]
                )
            chi_sq_lim *= n_bands - 1
        chi_sq = gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].chi_sq
        selected = chi_sq < chi_sq_lim
        kwargs_out = {"chi_sq": chi_sq, "n_bands": n_bands} if self.kwargs["reduced"] else {"chi_sq": chi_sq}
        return selected, kwargs_out


class Chi_Sq_Diff_Selector(SED_fit_Selector):
    """Select galaxies whose low-redshift solution is a poor fit.

    Compares the chi-squared of the free-redshift SED fit to the
    chi-squared of the best applicable low-redshift-capped ("lowz")
    fit (from `lowz_zmax_arr`, restricted to those with
    `zmax < zfree - dz`) and selects galaxies where the low-z fit is
    sufficiently worse, has no valid solution, or has no low-z run to
    compare against.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run whose free-redshift and low-z fits this
        selector compares.
    chi_sq_diff : `int` or `float`
        Minimum chi-squared difference (low-z minus free-z) required
        for selection.
    dz : `int` or `float`
        Minimum redshift offset between the free-redshift solution and
        an applicable low-z run's `zmax`.
    lowz_zmax_arr : `list` of `float`
        Sorted (ascending), candidate `zmax` values of the low-z SED
        fitting runs to compare against.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        chi_sq_diff: Union[int, float],
        dz: Union[int, float],
        lowz_zmax_arr: List[float],
    ):
        kwargs = {
            "chi_sq_diff": chi_sq_diff,
            "dz": dz,
            "lowz_zmax_arr": lowz_zmax_arr,
        }
        super().__init__(aper_diam, SED_fitter, **kwargs)

    @property
    def _selection_name(self) -> str:
        chi_sq_name = f"dchi>{self.kwargs['chi_sq_diff']:.1f}" #f"chi_sq_diff>{self.kwargs['chi_sq_diff']:.1f}"
        dz_name = f"dz>{self.kwargs['dz']:.1f}"
        lowz_zmax_name = f"zmax=({','.join([f'{zmax:.1f}' for zmax in self.kwargs['lowz_zmax_arr']])})"
        return f"{chi_sq_name},{dz_name},{lowz_zmax_name}"
    
    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_diff", "dz", "lowz_zmax_arr"]
    
    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic low-z comparison kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["lowz_label", "z_lowz", "chisq_lowz"]` and
            `[str, float, float]`.
        """
        return ["lowz_label", "z_lowz", "chisq_lowz"], [str, float, float]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["chi_sq_diff"], (int, float))])
            assertions.extend([isinstance(self.kwargs["dz"], (int, float))])
            assertions.extend([self.kwargs["dz"] > 0.0])
            assertions.extend([self.kwargs["chi_sq_diff"] >= 0.0])
            assertions.extend([isinstance(self.kwargs["lowz_zmax_arr"], (list, np.ndarray))])
            assertions.extend([all(isinstance(zmax, (int, float)) for zmax in self.kwargs["lowz_zmax_arr"])])
            assertions.extend([all(zmax > 0.0 for zmax in self.kwargs["lowz_zmax_arr"])])
            # ensure lowz_zmax_arr is sorted in ascending order
            assertions.extend([
                all(self.kwargs["lowz_zmax_arr"][i] < self.kwargs["lowz_zmax_arr"][i + 1] \
                for i in range(len(self.kwargs["lowz_zmax_arr"]) - 1))
            ])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        try:
            assertions = []
            assertions.extend([
                self.SED_fitter.label in 
                gal.aper_phot[self.aper_diam].SED_results.keys()
            ])
            assertions.extend([
                all(f"{lowz_zmax:.1f}" in gal.aper_phot[self.aper_diam].\
                SED_results[self.SED_fitter.label].lowz_zmax_properties.keys() \
                for lowz_zmax in self.kwargs["lowz_zmax_arr"])
            ])
            #gal_SED_fit_labels = self._get_lowz_SED_fit_labels(gal)
            #assertions.extend([len(gal_SED_fit_labels) > 0])
            failed = not all(assertions)
        except:
            failed = True
        return failed
    
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        # extract redshift + chi_sq of zfree run
        zfree = gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].z
        chi_sq_zfree = gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].chi_sq
        # extract redshift and chi_sq of lowz runs
        # lowz_SED_fit_labels = [i for i in filter( \
        #     lambda label: float(label.split("zmax=")[-1][:3]) \
        #     < zfree - self.kwargs["dz"], \
        #     self._get_lowz_SED_fit_labels(gal)
        # )]
        lowz_SED_fit_labels = [
            i for i in filter(
                lambda label: \
                float(label) < zfree - self.kwargs["dz"], \
            self._get_lowz_SED_fit_labels(gal)
        )]
        kwargs_out = {}
        # if no lowz runs, do not select galaxy
        if len(lowz_SED_fit_labels) == 0:
            selected = False
            kwargs_out["lowz_label"] = None
            kwargs_out["z_lowz"] = None
            kwargs_out["chisq_lowz"] = None
        else:
            # sort the lowz_SED_fit_labels to get highest redshift applicable lowz run
            # lowz_SED_fit_label = sorted(lowz_SED_fit_labels, reverse = True, \
            #     key = lambda label: float(label.split("zmax=")[-1][:3]))[0]
            lowz_SED_fit_label = sorted(lowz_SED_fit_labels, reverse = True, \
                key = lambda label: float(label))[0]
            lowz_properties = gal.aper_phot[self.aper_diam]. \
                SED_results[self.SED_fitter.label].lowz_zmax_properties[lowz_SED_fit_label]
            z_lowz = lowz_properties["zbest"]
            chi_sq_lowz = lowz_properties["chi2_best"]
            selected = (
                (chi_sq_lowz - chi_sq_zfree > self.kwargs["chi_sq_diff"])
                or (chi_sq_lowz == -1.0)
                or (z_lowz < 0.0)
            )
            kwargs_out["lowz_label"] = lowz_SED_fit_label
            kwargs_out["z_lowz"] = z_lowz
            kwargs_out["chisq_lowz"] = chi_sq_lowz
        return selected, kwargs_out

    def _get_lowz_SED_fit_labels(
        self: Self,
        gal: Galaxy,
    ) -> List[str]:
        # TODO: Works for EAZY, but not for a general 
        # SED fitting code with different zmax syntax
        return [
            label for label in getattr(
                gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label], \
                "lowz_zmax_properties",
                {}
            ).keys()
        ]
        # if "zmax=" in label and label.replace( \
        # f"_zmax={label.split('zmax=')[-1][:3]}", "") \
        # in self.SED_fitter.label
    
    # def _assert_cat(self: Self, cat: Catalogue) -> None:
    #     # ensure a lowz run has been run for at least 1 galaxy in the catalogue
    #     cat_SED_fit_labels = [self._get_lowz_SED_fit_labels(gal) for gal in cat]
    #     assert any(len(gal_labels) > 0 for gal_labels in cat_SED_fit_labels), \
    #         galfind_logger.critical(
    #             f"{repr(self.SED_fitter)} lowz not run for any galaxy."
    #         )
    #     super()._assert_cat(cat)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Optional[Catalogue]:
        self.SED_fitter._update_lowz_zmax(cat, self.aper_diam, self.kwargs["lowz_zmax_arr"])
        return SED_fit_Selector._call_cat(self, cat, return_copy)


class Chi_Sq_Template_Diff_Selector(SED_fit_Selector):
    """Select galaxies fit significantly better by one SED run than another.

    Compares the (optionally reduced) chi-squared of `SED_fitter` to
    that of `secondary_SED_fit_label` (e.g. a stellar/QSO template fit)
    and selects galaxies where the secondary run's chi-squared exceeds
    the primary's by more than `chi_sq_diff`, or is invalid.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        Primary SED-fitting code/run.
    chi_sq_diff : `int` or `float`
        Minimum chi-squared difference (secondary minus primary)
        required for selection.
    secondary_SED_fit_label : `str` or `SED_code`
        Label (or `SED_code` instance, whose `label` is used) of the
        secondary SED fitting run to compare against.
    reduced : `bool`, optional
        If `True`, compare reduced chi-squared values (accounting for
        differing numbers of degrees of freedom) rather than absolute
        chi-squared. Default is `False`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        chi_sq_diff: Union[int, float],
        secondary_SED_fit_label: Union[str, SED_code],
        reduced: bool = False,
    ):
        if isinstance(secondary_SED_fit_label, funcs.all_subclasses(SED_code)):
            secondary_SED_fit_label = secondary_SED_fit_label.label
        kwargs = {
            "chi_sq_diff": chi_sq_diff,
            "secondary_SED_fit_label": secondary_SED_fit_label,
            "reduced": reduced,
        }
        super().__init__(aper_diam, SED_fitter, **kwargs)

    @property
    def _selection_name(self) -> str:
        name = f"chi_sq_{self.kwargs['secondary_SED_fit_label']}>{self.kwargs['chi_sq_diff']:.1f}"
        if self.kwargs["reduced"]:
            name = f"red_{name}"
        return name
    
    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_diff", "secondary_SED_fit_label", "reduced"]
    
    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic comparison kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["chi_sq_1", "chi_sq_2"]` and `[float, float]`, plus
            `["n_bands_1", "n_bands_2"]`/`[int, int]` if `reduced` is
            `True`.
        """
        kwarg_names = ["chi_sq_1", "chi_sq_2"]
        kwarg_dtypes = [float, float]
        if self.kwargs["reduced"]:
            kwarg_names.extend(["n_bands_1", "n_bands_2"])
            kwarg_dtypes.extend([int, int])
        return kwarg_names, kwarg_dtypes

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
        *args,
        **kwargs,
    ) -> bool:
        try:
            failed = not all(
                label in gal.aper_phot[self.aper_diam].SED_results.keys()
                for label in [self.SED_fitter.label, self.kwargs["secondary_SED_fit_label"]]
            )
        except:
            failed = True
        return failed
    
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        # extract chi_sq/red_chi_sq of SED_fitting runs that are to be compared
        chi_sq = np.zeros(2)
        kwargs_out = {}
        for i, label in enumerate([self.SED_fitter.label, self.kwargs["secondary_SED_fit_label"]]):
            if not any(hasattr(gal.aper_phot[self.aper_diam].SED_results[label], chi_sq_label) for chi_sq_label in ["chi_sq", "red_chi_sq"]):
                for i in range(2):
                    kwargs_out[f"chi_sq_{i+1}"] = None
                    if self.kwargs["reduced"]:
                        kwargs_out[f"n_bands_{i+1}"] = 0
                return False
            if self.kwargs["reduced"]:
                if hasattr(gal.aper_phot[self.aper_diam].SED_results[label], "red_chi_sq"):
                    chi_sq[i] = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "red_chi_sq")
                    kwargs_out[f"n_bands_{i+1}"] = 0
                else:
                    chi_sq_ = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "chi_sq")
                    # work out number of degrees of freedom
                    if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
                        n_bands = len(gal.aper_phot[self.aper_diam].filterset.filt_names)
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
                    kwargs_out[f"n_bands_{i+1}"] = n_bands
            else: # absolute chi squared
                if hasattr(gal.aper_phot[self.aper_diam].SED_results[label], "chi_sq"):
                    chi_sq[i] = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "chi_sq")
                    kwargs_out[f"n_bands_{i+1}"] = None
                else:
                    chi_sq_ = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "red_chi_sq")
                    # work out number of degrees of freedom
                    if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
                        n_bands = len(gal.aper_phot[self.aper_diam].filterset.filt_names)
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
                    kwargs_out[f"n_bands_{i+1}"] = n_bands
            kwargs_out[f"chi_sq_{i+1}"] = chi_sq[i]
        selected = (
            (chi_sq[1] - chi_sq[0] > self.kwargs["chi_sq_diff"])
            or (chi_sq[1] < 0.0)
        )
        return selected, kwargs_out

    def _assert_cat(self: Self, cat: Catalogue) -> None:
        # ensure a secondary SED fitting run has been run for at least 1 galaxy in the catalogue
        cat_SED_fit_labels = [gal.aper_phot[self.aper_diam].SED_results.keys() for gal in cat]
        assert any(self.kwargs["secondary_SED_fit_label"] in gal_labels for gal_labels in cat_SED_fit_labels), \
            galfind_logger.critical(
                f"{self.kwargs['secondary_SED_fit_label']} not run for any galaxy."
            )
        super()._assert_cat(cat)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Catalogue]:
        return SED_fit_Selector._call_cat(self, cat, return_copy, *args, **kwargs)


class zPDF_High_Tail_Selector(SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        z_thr: float,
        p_lim: float,
    ):
        kwargs = {
            "z_thr": z_thr,
            "p_lim": p_lim,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        name = (
            f"P(z>{self.kwargs['z_thr']})>"
            f"{float(self.kwargs['p_lim'])}"
        )
        return name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["z_thr", "p_lim"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            # z_thr must be a positive number
            assertions.extend([isinstance(self.kwargs["z_thr"], (int, float))])
            assertions.extend([self.kwargs["z_thr"] > 0.0])
            # p_lim must be between 0 and 1
            assertions.extend([isinstance(self.kwargs["p_lim"], float)])
            assertions.extend([0.0 < self.kwargs["p_lim"] < 1.0])

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
        """
        Select galaxies with integrated PDF tail P(z > z_thr) > p_lim.
        """
        PDF = gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].property_PDFs["z"]

        # integrate P(z > z_thr)
        p_tail = PDF.integrate_between_lims(
            float(self.kwargs["z_thr"]),
            25.0,  # upper limit in galfind base code
        )

        return p_tail > self.kwargs["p_lim"]


class Robust_zPDF_Selector(SED_fit_Selector):
    """Select galaxies with a sufficiently unimodal/concentrated redshift PDF.

    Requires the fraction of the redshift PDF integrated within
    `dz_over_z` (or `min_dz`, if larger) of the best-fit redshift to
    exceed `integral_lim`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the SED fitting was performed at.
    SED_fitter : `SED_code`
        SED-fitting code/run whose redshift PDF this selector acts on.
    integral_lim : `float`
        Minimum fraction of the redshift PDF required within the
        window around the best-fit redshift.
    dz_over_z : `int` or `float`
        Fractional redshift window half-width, `dz / z`, used to
        integrate the PDF around the best-fit redshift.
    min_dz : `int` or `float`, optional
        Minimum absolute redshift window half-width; if
        `zbest * dz_over_z` is smaller than this, `dz_over_z` is
        scaled up so the window is at least `min_dz` wide. Default is
        `0.0`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        integral_lim: float,
        dz_over_z: Union[int, float],
        min_dz: Union[int, float] = 0.0,
    ):
        kwargs = {
            "integral_lim": integral_lim,
            "dz_over_z": dz_over_z,
            "min_dz": min_dz,
        }
        super().__init__(aper_diam, SED_fitter, **kwargs)

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

    @property
    def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
        """`tuple` of `list`: Names/dtypes of the diagnostic redshift-PDF kwargs.

        Returns
        -------
        `tuple` of (`list` of `str`, `list` of `type`)
            `["dz_z", "integral"]` and `[float, float]`.
        """
        return ["dz_z", "integral"], [float, float]

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
        *args,
        **kwargs,
    ) -> bool:
        try:
            failed = gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].z < 0.0
        except:
            failed = True
        return failed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        # extract best fitting redshift - peak of the redshift PDF
        zbest = gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label].z
        dz_z = self.kwargs["dz_over_z"]
        # adjust dz_z if zbest * dz_z is less than min_dz
        if zbest * dz_z < self.kwargs["min_dz"]:
            dz_z = self.kwargs["min_dz"] / zbest
        integral = gal.aper_phot[self.aper_diam].SED_results[self.SED_fitter.label]. \
            property_PDFs["z"].integrate_between_lims(
                float(dz_z),
                float(zbest),
            )
        selected = integral > self.kwargs["integral_lim"]
        kwargs_out = {"dz_z": dz_z, "integral": integral}
        return selected, kwargs_out


class Sextractor_Band_Radius_Selector(Data_Selector):
    """Select galaxies by SExtractor-measured effective radius in one band.

    Parameters
    ----------
    filt_name : `str`
        Name of the band the SExtractor radius is measured in.
    gtr_or_less : `str`
        Either ``"gtr"`` to select `Re > lim`, or ``"less"`` to select
        `Re < lim`.
    lim : `astropy.units.Quantity`
        Effective radius limit to compare against.
    """

    def __init__(
        self: Self,
        filt_name: str,
        gtr_or_less: str,
        lim: u.Quantity,
    ):
        kwargs = {
            "filt_name": filt_name,
            "gtr_or_less": gtr_or_less,
            "lim": lim,
        }
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        lim_str = f"{self.kwargs['lim'].to(u.marcsec).value:.1f}mas"
        gtr_or_less_str = ">" if self.kwargs["gtr_or_less"].lower() == "gtr" else "<"
        return f"sex_Re_{self.kwargs['filt_name']}{gtr_or_less_str}{lim_str}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["filt_name", "gtr_or_less", "lim"]

    def _assertions(self: Self) -> bool:
        try:
            self.kwargs["lim"].to(u.marcsec)
            assertions = []
            assertions.extend([isinstance(self.kwargs["filt_name"], str)])
            assertions.extend([self.kwargs["filt_name"] in json.loads(config.get("Other", "ALL_BANDS"))])
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
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        if self.kwargs["gtr_or_less"].lower() == "gtr":
            selected = gal.sex_Re[self.kwargs["filt_name"]] > self.kwargs["lim"]
        else: # self.kwargs["gtr_or_less"].lower() == "less"
            selected = gal.sex_Re[self.kwargs["filt_name"]] < self.kwargs["lim"]
        return selected, {}

    def _assert_cat(
        self: Self,
        cat: Catalogue,
    ) -> None:
        if isinstance(self.kwargs["filt_name"], str):
            assert (self.kwargs["filt_name"] in cat.filterset.filt_names), \
                galfind_logger.critical(
                    f"{self.kwargs['filt_name']} not in" + \
                    f" {cat.filterset.filt_names}!"
                )
        super()._assert_cat(cat)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Optional[Catalogue]:
        # load in effective radii as calculated from SExtractor
        cat.load_sextractor_Re()
        return Data_Selector._call_cat(self, cat, return_copy)


class Re_Selector(Morphology_Selector):
    """Select galaxies by morphology-fit effective radius.

    Parameters
    ----------
    morph_fitter : `Type`[`Morphology_Fitter`]
        Morphology-fitting code/run whose effective radius this
        selector acts on.
    gtr_or_less : `str`
        Either ``"gtr"`` to select `Re > lim`, or ``"less"`` to select
        `Re < lim`.
    lim : `astropy.units.Quantity`
        Effective radius limit to compare against.
    """

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
    
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        re_pix = gal.cutouts[self._cutout_str].morph_fits[self.morph_fitter.name].r_e
        if np.isnan(re_pix):
            return False, {}
        pix_scale = self.morph_fitter.psf.cutout.band_data.pix_scale / u.pix
        re_as = re_pix * pix_scale
        if self.kwargs["gtr_or_less"].lower() == "gtr":
            return re_as > self.kwargs["lim"], {}
        else: # self.kwargs["gtr_or_less"].lower() == "less"
            return re_as < self.kwargs["lim"], {}
    
class Kokorev24_LRD_red1_Selector(Multiple_Photometry_Selector):
    """Little Red Dot colour selection, criterion 1, from Kokorev et al. (2024).

    Combines F115W-F150W < 0.8, F200W-F277W > 0.7, and
    F200W-F356W > 1.0 colour cuts.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry was extracted at.
    """

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
    """Little Red Dot colour selection, criterion 2, from Kokorev et al. (2024).

    Combines F150W-F200W < 0.8, F277W-F356W > 0.6, and
    F277W-F444W > 0.7 colour cuts.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry was extracted at.
    """

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
    """Combined Little Red Dot colour selection from Kokorev et al. (2024).

    Requires both `Kokorev24_LRD_red1_Selector` and
    `Kokorev24_LRD_red2_Selector` to pass.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter the photometry was extracted at.
    """

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
    """Select unmasked pixels across all bands."""

    def __init__(
        self: Self, 
        filt_names: Union[str, List[str]]
    ):
        if isinstance(filt_names, str):
            filt_names = filt_names.split("+")
        selectors = [Unmasked_Band_Selector(filt_name = name) for name in filt_names]
        super().__init__(selectors, f"unmasked_{'+'.join(filt_names)}")
    
    def _assert_cat(self: Self, cat: Catalogue) -> None:
        assert all(name in cat.filterset.filt_names for name in \
                [selector.kwargs["filt_name"] for selector in self.selectors]), \
            galfind_logger.critical(
                f"Not all bands in {cat.filterset.filt_names}!"
            )
        super()._assert_cat(cat)


class Unmasked_Instrument_Selector(Multiple_Mask_Selector):
    """Select unmasked pixels within an instrument."""

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
        selectors = [Unmasked_Band_Selector(filt_name = name) for name in instrument.filt_names]
        super().__init__(selectors, f"unmasked_{instrument.__class__.__name__}", cat_filterset = cat_filterset)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Optional[Catalogue]:
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
    """Select sources by SExtractor radius across bands."""

    def __init__(
        self: Self,
        filt_names: List[str],
        gtr_or_less: str,
        lim: u.Quantity
    ):
        selectors = [
            Sextractor_Band_Radius_Selector(
                filt_name = filt_name,
                gtr_or_less = gtr_or_less,
                lim = lim
            ) for filt_name in filt_names
        ]
        lim_str = f"{lim.to(u.marcsec).value:.1f}mas"
        gtr_or_less_str = ">" if gtr_or_less.lower() == "gtr" else "<"
        selection_name = f"sex_Re_{'+'.join(filt_names)}{gtr_or_less_str}{lim_str}"
        super().__init__(selectors, selection_name)


class Sextractor_Instrument_Radius_Selector(Multiple_Data_Selector):
    """Select sources by SExtractor radius within an instrument."""

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
            instrument = [
                instr() for instr in Instrument.__subclasses__()
                if instr.__name__ == instrument
            ][0]
        selectors = [
            Sextractor_Band_Radius_Selector(
                filt_name = filt_name,
                gtr_or_less = gtr_or_less,
                lim = lim,
            ) for filt_name in instrument.filt_names
        ]
        lim_str = f"{lim.to(u.marcsec).value:.1f}mas"
        gtr_or_less_str = ">" if gtr_or_less.lower() == "gtr" else "<"
        selection_name = f"sex_Re_{instrument.__class__.__name__}{gtr_or_less_str}{lim_str}"
        super().__init__(selectors, selection_name, cat_filterset = cat_filterset)
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Optional[Catalogue]:
        self.crop_to_filterset(cat.filterset)
        return Multiple_Data_Selector._call_cat(self, cat, return_copy)


class Sextractor_Instrument_Radius_PSF_FWHM_Selector(Multiple_Data_Selector):
    """Select sources by SExtractor radius and PSF FWHM."""

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
        selectors = [
            Sextractor_Band_Radius_Selector(
                filt_name = filt_name,
                gtr_or_less = gtr_or_less,
                lim = fwhm_nircam[filt_name] * scaling,
            ) for filt_name in instrument.filt_names if filt_name in fwhm_nircam.keys()
        ]
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
    ) -> Optional[Catalogue]:
        self.crop_to_filterset(cat.filterset)
        assert all([filt.filt_name in fwhm_nircam.keys() for filt in cat.filterset if filt.instrument_name == "NIRCam"]), \
            galfind_logger.critical(
                f"{cat.filterset.instrument_name=} must have all NIRCam bands."
            )
        return Multiple_Data_Selector._call_cat(self, cat, return_copy)

    
class Brown_Dwarf_Selector(Multiple_SED_fit_Selector):
    """Select brown dwarf candidates from SED fits."""

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        red_chi_sq_diff: Union[int, float],
        gal_SED_fitter: Union[str, SED_code],
        ref_filt: str,
        compactness_radii: u.Quantity = [0.16, 0.32] * u.arcsec,
        compactness_lim: float = 1.1,
        compactness_psf_normed: bool = True,
        SNR_lim: Optional[Union[int, float]] = None,
        red_chi_sq_lim: Optional[Union[int, float]] = None,
    ):
        selectors = [
            Chi_Sq_Template_Diff_Selector(
                aper_diam,
                SED_fitter,
                chi_sq_diff = red_chi_sq_diff,
                secondary_SED_fit_label = gal_SED_fitter,
                reduced = True,
            ),
            Compactness_Selector(
                ref_filt,
                compare_radii = compactness_radii,
                compactness_lim = compactness_lim,
                psf_normed = compactness_psf_normed,
            ),
            Unmasked_Band_Selector(ref_filt),
        ]
        if isinstance(gal_SED_fitter, str):
            gal_SED_fitter_label = gal_SED_fitter
        else:
            gal_SED_fitter_label = gal_SED_fitter.label
        gal_SED_fitter_label = gal_SED_fitter_label.replace("_zfree", "")
        selection_name = f"dchi2_{gal_SED_fitter_label}>{red_chi_sq_diff:.1f}_{ref_filt}"
        if SNR_lim is not None:
            selectors.append(
                Band_SNR_Selector(
                    aper_diam,
                    band = ref_filt,
                    detect_or_non_detect = "detect",
                    SNR_lim = SNR_lim,
                )
            )
            selection_name += f"_SNR>{SNR_lim:.1f}"
        if red_chi_sq_lim is not None:
            selectors.append(
                Chi_Sq_Lim_Selector(
                    aper_diam,
                    SED_fitter,
                    chi_sq_lim = red_chi_sq_lim,
                    reduced = True,
                )
            )
            selection_name += f"_rchi2<{red_chi_sq_lim:.1f}"
        super().__init__(
            aper_diam,
            SED_fitter,
            selectors,
            selection_name = selection_name,
        )
    
    def __call__(self, cat, return_copy = True, *args, **kwargs):
        if "band_data" not in kwargs.keys():
            kwargs["band_data"] = cat.data.native[self.selectors[1].kwargs["filt_name"]]
        return super().__call__(cat, return_copy, *args, **kwargs)


class Hainline24_TY_Brown_Dwarf_Selector_1(Multiple_Photometry_Selector):
    """Select T/Y brown dwarfs using Hainline+24 criteria (method 1)."""

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
    """Select T/Y brown dwarfs using Hainline+24 criteria (method 2)."""

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


class EPOCHS_unmasked_criteria(Multiple_Mask_Selector):
    """Select unmasked pixels meeting EPOCHS survey criteria."""

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        forced_phot_band: List[str] = ["F277W", "F356W", "F444W"],
        fit_filterset: Optional[Multiple_Filter] = None,
        # allow_lowz: bool = False, # not used
    ):
        selectors = [
            Min_Instrument_Unmasked_Band_Selector(min_bands = 4, instrument = "NIRCam"), # unmasked in at least 4 NIRCam bands
            Unmasked_Bluewards_Lya_Selector(aper_diam, SED_fitter, widebands_only = False, n_bands = 1, fit_filterset = fit_filterset), # unmasked in the first band bluewards of the break
            Unmasked_Redwards_Lya_Selector(aper_diam, SED_fitter, widebands_only = True, n_bands = 2, fit_filterset = fit_filterset), # unmasked in the first 2 widebands redwards of the break
            #Unmasked_Redwards_Lya_Selector(aper_diam, SED_fitter, widebands_only = True, n_bands = "all", fit_filterset = fit_filterset), # unmasked in all widebands redwards of the break
        ]
        # must be unmasked in all forced photometry bands
        selectors.extend([Unmasked_Band_Selector(band) for band in forced_phot_band])
        selection_name = f"EPOCHS_unmasked_{SED_fitter.label}_{aper_diam.to(u.arcsec).value:.2f}as"
        super().__init__(selectors, selection_name = selection_name)


class EPOCHS_Selector(Multiple_SED_fit_Selector):
    """Select sources based on EPOCHS survey-specific SED criteria."""

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
        #allow_lowz: bool = False,
        simulated: bool = False,
        forced_phot_band: List[str] = ["F277W", "F356W", "F444W"],
        fit_filterset: Optional[Multiple_Filter] = None,
        sample: str = "robust",
    ):
        assert sample in ["good", "robust"], \
            galfind_logger.critical(
                f"{sample=} not valid. Must be 'good' or 'robust'."
            )
        if sample == "robust":
            chi_sq_lim = 3.0
        else: # sample == "good"
            chi_sq_lim = 6.0
        selectors = [
            Bluewards_Lya_Non_Detect_Selector(aper_diam, SED_fitter, SNR_lim = 2.0, fit_filterset = fit_filterset),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fitter, SNR_lims = [5.0, 5.0], widebands_only = True, fit_filterset = fit_filterset),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fitter, SNR_lims = 2.0, widebands_only = True, fit_filterset = fit_filterset),
            Chi_Sq_Lim_Selector(aper_diam, SED_fitter, chi_sq_lim = chi_sq_lim, reduced = True),
            Chi_Sq_Diff_Selector(aper_diam, SED_fitter, chi_sq_diff = 4.0, dz = 0.5, lowz_zmax_arr = [4.0, 6.0]),
            Robust_zPDF_Selector(aper_diam, SED_fitter, integral_lim = 0.6, dz_over_z = 0.1),
        ]
        # # add 2σ non-detection in first band if wanted
        # if not allow_lowz:
        #     selectors.extend([Band_SNR_Selector( \
        #         aper_diam, band = 0, SNR_lim = 2.0, detect_or_non_detect = "non_detect")])

        if not simulated:
            # add unmasked instrument selections
            selectors.extend([
                EPOCHS_unmasked_criteria(
                    aper_diam,
                    SED_fitter,
                    forced_phot_band,
                    fit_filterset = fit_filterset,
                )]) # lowz here
            # add hot pixel checks in forced photometry bands
            selectors.extend([
                Sextractor_Bands_Radius_Selector(
                    filt_names = forced_phot_band,
                    gtr_or_less = "gtr",
                    lim = 45. * u.marcsec
                )
            ])
        #lowz_name = "_lowz" if allow_lowz else ""
        selection_name = f"EPOCHS_{sample}" #{lowz_name}"
        super().__init__(aper_diam, SED_fitter, selectors, selection_name = selection_name)


class COSMOS_Web_Selector(Multiple_SED_fit_Selector):

    def __init__(
        self: Self,
        band_names: Union[str, List[str]],
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        allow_lowz: bool = False,
        unmasked_instruments: Union[str, List[str]] = "NIRCam", # TODO: Update this to allow for custom masking or change to default "EPOCHS" masking
        cat_filterset: Optional[Multiple_Filter] = None,
        simulated: bool = False,
    ):
        selectors = [
            Unmasked_Bands_Selector(band_names),
            Redshift_Limit_Selector(aper_diam, SED_fit_label, z_lim=8.5, gtr_or_less="gtr"),
            Bluewards_Lya_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim = 2.0),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = [10.0, 10.0], widebands_only = True),
            #Band_SNR_Selector(aper_diam, band = 'F150W', SNR_lim = 1.0, detect_or_non_detect = "detect"),
            #Robust_zPDF_Selector(aper_diam, SED_fit_label, integral_lim = 0.6, dz_over_z = 0.1),
            zPDF_High_Tail_Selector(aper_diam, SED_fit_label, z_thr=8.0, p_lim=0.8),
            Chi_Sq_Lim_Selector(aper_diam, SED_fit_label, chi_sq_lim = 8.0, reduced = False),
            Chi_Sq_Diff_Selector(aper_diam, SED_fit_label, chi_sq_diff = 5.0, dz = 0.5),
            Colour_Selector(aper_diam, colour_bands = ["F277W", "F444W"], bluer_or_redder = "bluer", colour_val = 1),
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
                band_names = ["F277W", "F444W"], \
                gtr_or_less = "gtr", lim = 45. * u.marcsec)
            ])
        lowz_name = "_lowz" if allow_lowz else ""
        unmasked_instr_name = "_" + "+".join(unmasked_instruments)
        selection_name = f"COSMOS-Web{lowz_name}{unmasked_instr_name}"
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name = selection_name)

# Catalogue Level Selection functions

# def select_all_bands(self):
#     return self.select_min_bands(len(self.instrument))

# Catalogue Level Selection functions

# def select_all_bands(self):
#     return self.select_min_bands(len(self.instrument))

class Rest_Frame_Property_Kwarg_Selector(SED_fit_Selector):
    """Select galaxies based on rest-frame property calculator keyword arguments.

    Selects galaxies by comparing specific keyword argument values used in
    rest-frame property calculations to a target value.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter for photometry selection.
    SED_fitter : `SED_code`
        SED code providing the SED results.
    property_calculator : `Type[Rest_Frame_Property_Calculator]`
        Property calculator class to extract keywords from.
    kwarg_name : `str`
        Name of the keyword argument to match.
    kwarg_val : `int` or `float`
        Target value for the keyword argument.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: SED_code,
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
        super().__init__(aper_diam, SED_fitter, **kwargs)

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
        *args,
        **kwargs,
    ) -> bool:
        if self.kwargs["kwarg_name"] not in gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fitter.label].phot_rest.property_kwargs \
                [self.kwargs["property_calculator"].name].keys():
            return True
        else:
            return False

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        return gal.aper_phot[self.aper_diam].SED_results \
            [self.SED_fitter.label].phot_rest.property_kwargs \
            [self.kwargs["property_calculator"].name] \
            [self.kwargs["kwarg_name"]] == self.kwargs["kwarg_val"], {}


# class Star_Selector(Data_Selector):

#     def __init__(
#         self: Self,
#         band_data: Band_Data,
#     ):
#         self.band_data = band_data
#         kwargs = {}
#         super().__init__(**kwargs)

#     @property
#     def _selection_name(self) -> str:
#         if self.kwargs["dz"] > 0.0:
#             dz_str = f"_dz{self.kwargs['dz']:.2f}"
#         else:
#             dz_str = ""
#         selection_name = f"blue_Lya_stacked_SNR<{self.kwargs['SNR_lim']:.1f}{dz_str}_zmax{self.kwargs['zmax']:.1f}"
#         return selection_name

#     @property
#     def _include_kwargs(self) -> List[str]:
#         return ["SNR_lim", "dz", "zmax"] # "filterset", "detect_or_non_detect", 

#     @property
#     def select_kwarg_names_dtypes(self: Self) -> Tuple[List[str], List[Type]]:
#         return ["bands", "SNR"], [str, float]

#     def _assertions(self: Self) -> bool:
#         try:
#             assertions = []
#             #assertions.extend([isinstance(self.kwargs["filterset"], Multiple_Filter)])
#             assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
#             assertions.extend([isinstance(self.kwargs["dz"], (int, float))])
#             assertions.extend([self.kwargs["dz"] >= 0.0])
#             assertions.extend([isinstance(self.kwargs["zmax"], (int, float))])
#             assertions.extend([self.kwargs["zmax"] > 0.0])
#             #assertions.extend([self.kwargs["detect_or_non_detect"].lower() in ["detect", "non_detect"]])
#             passed = all(assertions)
#         except:
#             passed = False
#         return passed

#     def _failure_criteria(
#         self: Self,
#         gal: Galaxy,
#         *args,
#         **kwargs,
#     ) -> bool:
#         blue_stack_label = self._get_blue_stack_label(gal)
#         if blue_stack_label is None:
#             return True
#         else:
#             return False
        
#     def _selection_criteria(
#         self: Self,
#         gal: Galaxy,
#         *args,
#         **kwargs
#     ) -> Tuple[bool, Dict[str, Any]]:
#         blue_stack_label = self._get_blue_stack_label(gal)
#         if not "+" in blue_stack_label:
#             SNR = float(gal.aper_phot[self.aper_diam][blue_stack_label].SNR[0].unmasked)
#         else:
#             SNR = float(gal.aper_phot[self.aper_diam].stacked_band_snrs[blue_stack_label])
#         if not np.isfinite(SNR):
#             selected = False
#         else:
#             selected = SNR < self.kwargs["SNR_lim"]
#         kwargs_out = {"bands": blue_stack_label, "SNR": SNR}
#         return selected, kwargs_out

#     def _assert_cat(
#         self: Self,
#         cat: Catalogue,
#     ) -> None:
#         req_stacked_bands = self._required_stacked_bands(cat.filterset)
#         stacked_band_data_filt_names = [band_data.filt_name for band_data in cat.data.stacked_band_data_arr]
#         if not all(["+".join(req_stacked_band.filt_names) in stacked_band_data_filt_names for req_stacked_band in req_stacked_bands]):
#             err_message = f"{repr(self)} requires the following stacked bands: " + \
#                 f"{['+'.join(req_stacked_band.filt_names) for req_stacked_band in req_stacked_bands]}, " + \
#                 f"but only the following stacked bands are available: {stacked_band_data_filt_names}!"
#             galfind_logger.critical(err_message)
#             raise Exception(err_message)
#         cat.load_stacked_band_data_snrs()
#         super()._assert_cat(cat)


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
    #                 self.phot.instrument.index_from_filt_name(band.filt_name)
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
    #         closest_band_index = self.phot.instrument.index_from_filt_name(
    #             closest_band.filt_name
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
    #                 self.phot.instrument.index_from_filt_name(band.filt_name)
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
    #             closest_band.filt_name,
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
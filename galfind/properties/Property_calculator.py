"""Derived galaxy property calculators with uncertainty propagation.

Provides abstract base class for computing derived galaxy properties (UV magnitude,
UV slope, stellar mass) from photometry/SED results with uncertainty propagation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
from galfind.catalogues.Catalogue import Catalogue
import numpy as np
from numpy.typing import NDArray
from astropy.table import Table
from copy import deepcopy
import astropy.units as u
from typing import TYPE_CHECKING, Dict, Any, List, Union, Tuple, Optional
if TYPE_CHECKING:
    from . import Catalogue, Galaxy, Photometry_rest, SED_code, SED_obs, PDF, Morphology_Fitter, Band_Cutout_Base
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger, all_filt_names, astropy_cosmo
from ..utils import useful_funcs_austind as funcs
from ..catalogues.Catalogue import Catalogue
from ..catalogues.Catalogue_Base import Catalogue_Base
from ..galaxy.Galaxy import Galaxy
from ..sed_fitting.SED_codes import SED_code
from ..sed_fitting.SED_result import SED_result
from ..visualization.PDF import PDF, SED_fit_PDF
from ..spectra.SED import SED_obs

class Property_Calculator_Base(ABC):
    """Abstract base class defining the calling contract for property calculators.

    Subclasses compute a derived galaxy or photometric property (e.g.
    absolute UV magnitude, UV continuum slope, stellar mass) from
    `Catalogue`, `Galaxy`, or `Photometry_rest` objects. Concrete
    subclasses must implement the `name`, `full_name`, and `plot_name`
    identifying properties, along with the `__call__`, `_call_cat`, and
    `_call_gal` dispatch methods that apply the calculation to a
    `Catalogue`, `Galaxy`, or single object respectively.
    """

    @property
    @abstractmethod
    def full_name(self: Self) -> str:
        """Unique string identifier for this property calculation.

        Returns
        -------
        `str`
            The fully-qualified property name, typically including any
            aperture-diameter or SED-fitting-code suffixes.
        """
        pass

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Short string identifier for this property.

        Returns
        -------
        `str`
            The property name, used as a dictionary/column key.
        """
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        """Human-readable label for this property.

        Returns
        -------
        `str`
            The plot label, potentially containing LaTeX math.
        """
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
        """Compute and apply this property calculation to an object.

        Dispatches to `_call_cat` or `_call_gal` depending on the type
        of `object`.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Photometry_rest`
            The object to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw when propagating
            uncertainties. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified object. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs to use when looping over a catalogue.
            Default is `1`.

        Returns
        -------
        `Catalogue`, `Galaxy`, `Photometry_rest`, or `None`
            The updated object, or `None`, depending on the implementation.
        """
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
        """Apply this property calculation to every galaxy in a catalogue.

        Parameters
        ----------
        cat : `Catalogue`
            The catalogue to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified catalogue. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs to use. Default is `1`.

        Returns
        -------
        `Catalogue` or `None`
            The updated catalogue, or `None`.
        """
        pass

    @abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_dir: str = "",
    ) -> Optional[Galaxy]:
        """Apply this property calculation to a single galaxy.

        Parameters
        ----------
        gal : `Galaxy`
            The galaxy to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified galaxy. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        save_dir : `str`, optional
            Directory to save intermediate outputs to. Default is `""`.

        Returns
        -------
        `Galaxy` or `None`
            The updated galaxy, or `None`.
        """
        pass


class Property_Calculator(Property_Calculator_Base):
    """Base class for property calculators tied to a fixed aperture diameter.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    **kwargs : `dict`
        Additional keyword arguments stored and passed through to the
        underlying property calculation.

    Attributes
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter associated with this calculator.
    kwargs : `dict`
        Extra keyword arguments supplied at construction.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.aper_diam = aper_diam
        self.kwargs = kwargs

    @property
    def full_name(self: Self) -> str:
        """Unique string identifier including the aperture diameter suffix.

        Returns
        -------
        `str`
            `"{name}_{aper_diam in arcsec:.2f}as"`.
        """
        return f"{self.name}_{self.aper_diam.to(u.arcsec).value:.2f}as"

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Short string identifier for this property.

        Returns
        -------
        `str`
            The property name, used as a dictionary/column key.
        """
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        """Human-readable label for this property.

        Returns
        -------
        `str`
            The plot label, potentially containing LaTeX math.
        """
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
        """Compute and apply this property calculation to an object.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Photometry_rest`
            The object to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified object. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs to use. Default is `1`.

        Returns
        -------
        `Catalogue`, `Galaxy`, `Photometry_rest`, or `None`
            The updated object, or `None`, depending on the implementation.
        """
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
        """Apply this property calculation to every galaxy in a catalogue.

        Parameters
        ----------
        cat : `Catalogue`
            The catalogue to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified catalogue. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs to use. Default is `1`.

        Returns
        -------
        `Catalogue` or `None`
            The updated catalogue, or `None`.
        """
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
        """Apply this property calculation to a single galaxy.

        Parameters
        ----------
        gal : `Galaxy`
            The galaxy to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified galaxy. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        save_dir : `str`, optional
            Directory to save intermediate outputs to. Default is `""`.

        Returns
        -------
        `Galaxy` or `None`
            The updated galaxy, or `None`.
        """
        pass

    # @abstractmethod
    # def _calculate(
    #     self: Self,
    #     object: Union[Catalogue, Galaxy, Photometry_rest, SED_result],
    # ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
    #     pass

#
class Property_Extractor:
    """Mixin forcing a property calculator to run without parallelism.

    Overrides the dispatch methods to call the next class in the method
    resolution order with a single job (`n_jobs = 1`). Intended to be
    combined with a `Property_Calculator`-derived class via multiple
    inheritance, for calculators that must run sequentially in a single
    process (e.g. because the underlying computation is not picklable).
    """

    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, SED_result],
        *args,
        **kwargs
        #save: bool = True,
    ) -> Optional[Catalogue]:
        # call the super with n_jobs = 1
        return super().__call__(object, *args, **kwargs)

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
    """Abstract base class for calculators deriving properties from observed-frame photometry.

    Concrete subclasses compute quantities such as colours, magnitudes, or
    signal-to-noise ratios directly from observed (not rest-frame or
    SED-fit) photometry.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    **kwargs : `dict`
        Additional keyword arguments stored and passed through to the
        underlying property calculation.
    """

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Short string identifier for this property.

        Returns
        -------
        `str`
            The property name, used as a dictionary/column key.
        """
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        """Human-readable label for this property.

        Returns
        -------
        `str`
            The plot label, potentially containing LaTeX math.
        """
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
        """Compute and apply this property calculation to an object.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Photometry_rest`
            The object to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified object. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs to use. Default is `1`.

        Returns
        -------
        `Catalogue`, `Galaxy`, `Photometry_rest`, or `None`
            The updated object, or `None`, depending on the implementation.
        """
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
        """Apply this property calculation to every galaxy in a catalogue.

        Parameters
        ----------
        cat : `Catalogue`
            The catalogue to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified catalogue. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs to use. Default is `1`.

        Returns
        -------
        `Catalogue` or `None`
            The updated catalogue, or `None`.
        """
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
        """Apply this property calculation to a single galaxy.

        Parameters
        ----------
        gal : `Galaxy`
            The galaxy to compute the property for.
        n_chains : `int`, optional
            Number of posterior samples/chains to draw. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified galaxy. Default is `False`.
        overwrite : `bool`, optional
            Whether to recompute and overwrite an already-stored value.
            Default is `False`.
        save_dir : `str`, optional
            Directory to save intermediate outputs to. Default is `""`.

        Returns
        -------
        `Galaxy` or `None`
            The updated galaxy, or `None`.
        """
        pass

class Photometry_Property_Loader(Photometry_Property_Calculator):
    """Loads or computes an observed-frame photometric property for every galaxy in a catalogue.

    Uses a user-supplied function to retrieve raw per-galaxy values (or
    posterior samples), then caches percentile summaries and PDFs so they
    can be attached to individual galaxies.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    property_getter : `callable`
        Function of the form ``property_getter(cat, aper_diam, **kwargs) ->
        (IDs, properties_arr)`` returning galaxy IDs and either a 1D array
        of property values or a 2D array of posterior samples (one row per
        galaxy).
    name : `str`
        Short identifier for the loaded property.
    plot_name : `str` or `None`
        Human-readable label for plotting.
    units : `astropy.units.Unit`
        Physical units of the loaded property.
    **kwargs : `dict`
        Additional keyword arguments passed through to `property_getter`.

    Attributes
    ----------
    units : `astropy.units.Unit`
        Units associated with the loaded property values.
    property_getter : `callable`
        Function used to retrieve the raw property values.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        property_getter: Callable[Tuple[Catalogue, u.Quantity], NDArray],
        name: str,
        plot_name: Optional[str],
        units: u.Unit,
        **kwargs: Dict[str, Any],
    ) -> None:
        self._name = name
        self._plot_name = plot_name
        self.units = units
        self.property_getter = property_getter
        super().__init__(aper_diam, **kwargs)

    @property
    def name(self: Self) -> str:
        """Short string identifier for this loaded property.

        Returns
        -------
        `str`
            The property name supplied at construction.
        """
        return self._name

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for this loaded property.

        Returns
        -------
        `str`
            The plot label supplied at construction.
        """
        return self._plot_name

    def _get_IDs_properties(
        self: Self,
        cat: Type[Catalogue_Base],
    ) -> Tuple[NDArray, NDArray]:
        IDs, properties_arr = self.property_getter(cat, self.aper_diam, **self.kwargs)
        self._IDs = IDs
        if len(properties_arr.shape) == 1:
            self._properties = np.vstack((np.zeros(len(properties_arr)), properties_arr, np.zeros(len(properties_arr))))
            # array of None
            self._property_PDFs = None #np.array(list(itertools.repeat(None, len(properties_arr))))
        else:
            self._properties = np.percentile(properties_arr, [16, 50, 84], axis = 1)
            self._property_PDFs = np.array([PDF.from_1D_arr(self.name, properties * self.units) if not all(np.isnan(properties)) else None for properties in properties_arr])
    
    def __call__(
        self: Self,
        object: Type[Galaxy, Catalogue_Base],
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> NDArray:
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            val = self._call_cat(object)
        else:
            val = self._call_gal(object)
            # breakpoint()
            # err_message = f"{object=} with {type(object)=} != Catalogue"
            # galfind_logger.critical(err_message)
            # raise TypeError(err_message)
        return val
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> NDArray:
        self._get_IDs_properties(cat)
        return np.array([self._call_gal(gal).value for gal in cat]) * self.units
    
    def _call_gal(self: Self, gal: Galaxy) -> NDArray:
        # load property into galaxy
        if not hasattr(gal.aper_phot[self.aper_diam], "properties"):
            gal.aper_phot[self.aper_diam].properties = {}
        if not hasattr(gal.aper_phot[self.aper_diam], "property_errs"):
            gal.aper_phot[self.aper_diam].property_errs = {}
        if not hasattr(gal.aper_phot[self.aper_diam], "property_PDFs"):
            gal.aper_phot[self.aper_diam].property_PDFs = {}

        gal.aper_phot[self.aper_diam].properties[self.name] = self._properties[1][self._IDs == f"{gal.ID}_{gal.survey}"][0] * self.units
        gal.aper_phot[self.aper_diam].property_errs[self.name] = np.array([self._properties[1][self._IDs == f"{gal.ID}_{gal.survey}"] - \
            self._properties[0][self._IDs == f"{gal.ID}_{gal.survey}"], self._properties[2][self._IDs == f"{gal.ID}_{gal.survey}"] - \
            self._properties[1][self._IDs == f"{gal.ID}_{gal.survey}"]]) * self.units

        if self._property_PDFs is None:
            gal.aper_phot[self.aper_diam].property_PDFs[self.name] = None
        else:
            pdf = self._property_PDFs[self._IDs == f"{gal.ID}_{gal.survey}"]
            try:
                if pdf is not None:
                    gal.aper_phot[self.aper_diam].property_PDFs[self.name] = pdf[0]
                else:
                    gal.aper_phot[self.aper_diam].property_PDFs[self.name] = None
            except:
                breakpoint()
        return gal.aper_phot[self.aper_diam].properties[self.name]
    
    def extract_vals(
        self: Self,
        object: Type[Catalogue_Base]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the median loaded property value for every galaxy in a catalogue.

        Parameters
        ----------
        object : `Catalogue`
            The catalogue (or `Catalogue_Base` subclass instance) to
            extract values from.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            Array of property values, one per galaxy, with consistent
            units attached.

        Raises
        ------
        AssertionError
            If the per-galaxy property units are not all consistent.
        TypeError
            If `object` is not a `Catalogue_Base` subclass instance.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            cat_vals = [gal.aper_phot[self.aper_diam].properties[self.name] for gal in object]
            if not all(isinstance(val, float) for val in cat_vals):
                assert all(val.unit == cat_vals[0].unit for val in cat_vals), \
                    galfind_logger.critical(f"Units of {self.name} in {object} are not consistent")
                cat_vals = np.array([val.value for val in cat_vals]) * cat_vals[0].unit
            else:
                cat_vals = np.array(cat_vals)
            return cat_vals
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, SED_obs]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

    # TODO: Propagate from parent class
    def extract_errs(
        self: Self,
        object: Type[Catalogue_Base]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the lower/upper loaded property uncertainties for every galaxy in a catalogue.

        Parameters
        ----------
        object : `Catalogue`
            The catalogue (or `Catalogue_Base` subclass instance) to
            extract uncertainties from.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            Array of (lower, upper) uncertainties, one pair per galaxy.

        Raises
        ------
        AssertionError
            If the per-galaxy property error units are not all consistent.
        NotImplementedError
            If `object` is not a `Catalogue_Base` subclass instance.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            cat_errs = [gal.aper_phot[self.aper_diam].property_errs[self.name] for gal in object]
            if not all(isinstance(val, float) for val in cat_errs):
                assert all(val.unit == cat_errs[0].unit for val in cat_errs), \
                    galfind_logger.critical(f"Units of {self.name} in {object} are not consistent")
                cat_errs = np.array([val.value for val in cat_errs]) * cat_errs[0].unit
            else:
                cat_errs = np.array(cat_errs)
            return cat_errs[:, :, 0]
        raise NotImplementedError()

    def extract_PDFs(
        self: Self,
        object: Type[Catalogue_Base],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the posterior PDF of the loaded property for every galaxy in a catalogue.

        Parameters
        ----------
        object : `Catalogue`
            The catalogue (or `Catalogue_Base` subclass instance) to
            extract PDFs from.

        Returns
        -------
        `list` of `PDF` or `None`
            One PDF (or `None` if unavailable) per galaxy.

        Raises
        ------
        TypeError
            If `object` is not a `Catalogue_Base` subclass instance.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            return [gal.aper_phot[self.aper_diam].property_PDFs[self.name] for gal in object]
        else:
            err_message = f"{object=} with {type(object)=} != Catalogue"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

class Band_SNR_Loader(Photometry_Property_Loader):
    """Loads the signal-to-noise ratio (SNR) in a single photometric band for every galaxy in a catalogue.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry to extract the SNR from.
    filt_name : `str`
        Name of the filter/band to load the SNR for.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        filt_name: str,
    ) -> None:
        _name = f"{filt_name}_SNR"
        _plot_name = f"{filt_name} SNR"
        units = u.dimensionless_unscaled
        kwargs = {"filt_name": filt_name}
        super().__init__(aper_diam, Band_SNR_Loader._load_SNR, _name, _plot_name, units, **kwargs)

    @staticmethod
    def _load_SNR(cat: Catalogue, aper_diam: u.Quantity, **kwargs) -> Tuple[NDArray, NDArray]:
        assert "filt_name" in kwargs.keys()
        # determine relevant band indices
        IDs = np.array([f"{gal.ID}_{gal.survey}" for gal in cat])
        band_SNRs = np.array(
            [[SNR for SNR, filt_name in zip(
                gal.aper_phot[aper_diam].SNR,
                gal.aper_phot[aper_diam].filterset.filt_names
            ) if filt_name == kwargs["filt_name"]] for gal in cat]
        ).flatten()
        return IDs, band_SNRs


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
    """Abstract base class for calculators deriving properties from morphological fits to image cutouts.

    Parameters
    ----------
    morph_fitter : `Morphology_Fitter`
        The morphology fitter whose fitted parameters this calculator
        extracts or derives from.

    Attributes
    ----------
    morph_fitter : `Morphology_Fitter`
        The associated morphology fitter instance.
    """

    def __init__(
        self: Self,
        morph_fitter: Morphology_Fitter
    ) -> None:
        self.morph_fitter = morph_fitter

    @property
    def cutout_label(self: Self) -> str:
        """Label identifying the filter band and cutout size used for the morphological fit.

        Returns
        -------
        `str`
            `"{filter name}_{cutout size in arcsec:.2f}as"`.
        """
        return self.morph_fitter.psf.cutout.band_data.filt_name + \
            f"_{self.morph_fitter.psf.cutout.cutout_size.to(u.arcsec).value:.2f}as"

    @property
    def full_name(self: Self) -> str:
        """Unique string identifier including the morphology fitter name.

        Returns
        -------
        `str`
            `"{name}_{morph_fitter.name}"`.
        """
        return f"{self.name}_{self.morph_fitter.name}"

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Short string identifier for this morphological property.

        Returns
        -------
        `str`
            The property name, used as a dictionary/column key.
        """
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        """Human-readable label for this morphological property.

        Returns
        -------
        `str`
            The plot label, potentially containing LaTeX math.
        """
        pass

    def __call__(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Type[Band_Cutout_Base]],
    ) -> Optional[Union[Type[Catalogue_Base], Galaxy, Type[Band_Cutout_Base]]]:
        from ..visualization.Cutout import Band_Cutout_Base
        from ..catalogues.Catalogue_Base import Catalogue_Base
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
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
        object: Union[Type[Catalogue_Base], Galaxy, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the fitted/derived morphological property value(s).

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Band_Cutout_Base`
            The object to extract the morphological property value(s) from.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            The property value (for a single galaxy/cutout), or an array
            of values (for a catalogue).

        Raises
        ------
        AssertionError
            If the per-galaxy property units are not all consistent
            (catalogue case).
        TypeError
            If `object` is not a `Catalogue_Base`, `Galaxy`, or
            `Band_Cutout_Base` subclass instance.
        """
        from . import Band_Cutout_Base
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
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
        object: Union[Type[Catalogue_Base], Galaxy, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the posterior PDF of the morphological property.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Band_Cutout_Base`
            The object to extract the PDF(s) from.

        Returns
        -------
        `PDF` or `list` of `PDF`
            The PDF for a single galaxy/cutout, or a list of PDFs for a
            catalogue.

        Raises
        ------
        TypeError
            If `object` is not a `Catalogue_Base`, `Galaxy`, or
            `Band_Cutout_Base` subclass instance.
        """
        from . import Band_Cutout_Base
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            return [gal.cutouts[self.cutout_label].morph_fits \
                [self.morph_fitter.name].property_pdfs[self.name] for gal in object]
        elif isinstance(object, Galaxy):
            return object.cutouts[self.cutout_label].morph_fits \
                [self.morph_fitter.name].property_pdfs[self.name]
        elif isinstance(object, tuple(Band_Cutout_Base.__subclasses__())):
            return object.property_pdfs[self.name]
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, Band_Cutout_Base]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)


class Custom_Morphology_Property_Extractor(Property_Extractor, Morphology_Property_Calculator):
    """Extracts an arbitrary named morphological fit parameter without recomputation.

    Parameters
    ----------
    name : `str`
        Short identifier of the morphological parameter to extract (e.g.
        an attribute name stored on the morphology fit result).
    plot_name : `str`
        Human-readable label for plotting.
    morph_fitter : `Morphology_Fitter`
        The morphology fitter whose fitted parameter this extracts.
    """

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
        """Short string identifier for this extracted property.

        Returns
        -------
        `str`
            The property name supplied at construction.
        """
        return self._name

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for this extracted property.

        Returns
        -------
        `str`
            The plot label supplied at construction.
        """
        return self._plot_name


# Calculates additional properties from best fitting rest frame SED
class SED_Property_Calculator(Property_Calculator):
    """Abstract base class for calculators deriving properties from the best-fitting rest-frame SED.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used in the SED fit.
    SED_fit_label : `str` or `Type[SED_code]`
        The SED fitting code (or its label) whose results this calculator uses.

    Attributes
    ----------
    SED_fit_label : `str`
        Label identifying the SED fitting run to use.
    """

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
        """Unique string identifier including the SED-fit label and aperture diameter.

        Returns
        -------
        `str`
            `"{name}_{SED_fit_label}_{aper_diam in arcsec:.2f}as"`.
        """
        return f"{self.name}_{self.SED_fit_label}" + \
            f"_{self.aper_diam.to(u.arcsec).value:.2f}as"

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Short string identifier for this SED-derived property.

        Returns
        -------
        `str`
            The property name, used as a dictionary/column key.
        """
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        """Human-readable label for this SED-derived property.

        Returns
        -------
        `str`
            The plot label, potentially containing LaTeX math.
        """
        pass

    def __call__(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
        *args,
        **kwargs
    ) -> Optional[Union[Type[Catalogue_Base], Galaxy, Photometry_rest]]:
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
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
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
        cat_properties = [
            self._call_gal(gal)
            for gal in tqdm(
                cat,
                desc = f"Calculating {repr(self)} for {repr(cat)}",
                total = len(cat),
                disable = galfind_logger.getEffectiveLevel() > logging.INFO
            )
        ]
        if isinstance(cat_properties[0], (u.Quantity, u.Magnitude, u.Dex)):
            cat_properties = np.array([prop.value for prop in cat_properties]) * cat_properties[0].unit
        # write to output table that is appropriately named
        self.update_fits_cat(cat, cat_properties)
        return cat

    def update_fits_cat(
        self: Self,
        cat: Catalogue,
        cat_properties: NDArray,
    ) -> None:
        """Write computed per-galaxy property values into the catalogue's FITS table.

        Adds a new column for this property to the appropriate
        ``SED_PROPERTIES_*`` HDU of the catalogue's FITS file, creating the
        HDU if it does not already exist. Does nothing if the column
        already exists.

        Parameters
        ----------
        cat : `Catalogue`
            The catalogue whose FITS file should be updated.
        cat_properties : `numpy.ndarray`
            Array of property values, one per galaxy in `cat`, in
            catalogue order.

        Returns
        -------
        `None`

        Raises
        ------
        AssertionError
            If the existing FITS table has a different number of rows than
            `cat`, or if `SED_fit_label` is missing from any galaxy's SED
            results.
        """
        # TODO: generalize this funciton further
        # determine appropriate hdu and name to save properties as
        # TODO: generalize hdu name for non-EAZY SED fitting labels
        property_hdu = f"SED_PROPERTIES_{'_'.join(self.SED_fit_label.split('_')[:-1])}".upper()
        property_name = f"{self.name}_{self.aper_diam.to(u.arcsec).value:.2f}as"
        # open fits catalogue
        tab = cat.open_cat(hdu=property_hdu)
        if tab is None:
            write = True
            tab = Table()
            tab["ID"] = np.array([gal.ID for gal in cat])
        elif not property_name in tab.colnames:
            write = True
            assert len(tab) == len(cat)
        else:
            write = False

        if write:
            assert all([self.SED_fit_label in gal.aper_phot[self.aper_diam].SED_results.keys() for gal in cat])
            # if all([self.SED_fit_label in gal.aper_phot[self.aper_diam].SED_results.keys() for gal in cat]):
            #     kwarg_names = np.unique([key for gal in cat for key in \
            #         gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label]. \
            #         phot_rest.property_kwargs[self.name].keys()])
            #     for kwarg_name in kwarg_names:
            #         tab[f"{property_name}_{kwarg_name}"] = [gal.aper_phot[self.aper_diam]. \
            #             SED_results[self.SED_fit_label].phot_rest.property_kwargs[self.name] \
            #             [kwarg_name] if kwarg_name in gal.aper_phot[self.aper_diam].SED_results \
            #             [self.SED_fit_label].phot_rest.property_kwargs[self.name].keys() else np.nan for gal in cat]
            # else:
            #     raise(
            #         ValueError(
            #             galfind_logger.critical(
            #                 f"{self.SED_fit_label=} not in all galaxies in {cat.survey} {cat.version}"
            #             )
            #         )
            #     )
            # update fits catalogue
            tab[property_name] = cat_properties
            cat.write_hdu(tab, hdu=property_hdu)

    def _call_gal(
        self: Self,
        gal: Galaxy,
        **kwargs,
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
        return self._call_sed_result(gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label], **kwargs)
        # if output:
        #     return gal
    
    def _call_sed_result(
        self: Self,
        sed_result: SED_result,
        **kwargs
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
        object: Union[Type[Catalogue_Base], Galaxy, SED_obs]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the computed SED-derived property value(s).

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `SED_obs`
            The object to extract the property value(s) from.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            The property value (for a single galaxy/`SED_obs`), or an
            array of values (for a catalogue).

        Raises
        ------
        AssertionError
            If the per-galaxy property units are not all consistent
            (catalogue case).
        TypeError
            If `object` is not a `Catalogue_Base` subclass, `Galaxy`, or
            `SED_obs` instance.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
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
        object: Union[Type[Catalogue_Base], Galaxy, SED_obs]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the property uncertainties for every galaxy in a catalogue.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `SED_obs`
            The object to extract uncertainties from. Only `Catalogue_Base`
            subclass instances are currently supported.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            The stored property uncertainties, one entry per galaxy.

        Raises
        ------
        NotImplementedError
            If `object` is not a `Catalogue_Base` subclass instance.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            try:
                cat_errs = [gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].property_errs[self.name] for gal in object]
            except:
                breakpoint()
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
        object: Union[Type[Catalogue_Base], Galaxy, SED_obs],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the posterior PDF of the SED-derived property.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `SED_obs`
            The object to extract the PDF(s) from.

        Returns
        -------
        `PDF` or `list` of `PDF`
            The PDF for a single galaxy/`SED_obs`, or a list of PDFs for a
            catalogue.

        Raises
        ------
        TypeError
            If `object` is not a `Catalogue_Base` subclass, `Galaxy`, or
            `SED_obs` instance.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
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


class MUV_SED_Property_Calculator(SED_Property_Calculator):
    """Computes the rest-frame absolute UV magnitude (M_UV) from the best-fit SED.

    Optionally applies an extended-source aperture correction to the
    computed magnitude, using the correction factor of the observed band
    nearest a given rest-frame reference wavelength.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used in the SED fit.
    SED_fitter : `Type[SED_code]`
        The SED fitting code whose best-fit SED is used to compute M_UV.
    ext_src_corrs : `str`, optional
        Extended-source correction to apply: either `"UV"` (uses the band
        nearest `ref_wav`) or an individual band name. If `None`, no
        correction is applied. Default is `"UV"`.
    ext_src_uplim : `int` or `float`, optional
        Upper limit imposed on the extended-source correction factor.
        Default is `10.0`.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength used to select the extended-source
        correction band when `ext_src_corrs == "UV"`. Default is
        `1_500.0 * astropy.units.AA`.

    Attributes
    ----------
    SED_fitter : `Type[SED_code]`
        The SED fitting code used to compute M_UV.
    ext_src_corrs : `str` or `None`
        Extended-source correction setting.
    ext_src_uplim : `int`, `float`, or `None`
        Upper limit on the extended-source correction factor.
    ref_wav : `astropy.units.Quantity`
        Reference wavelength used for extended-source correction band
        selection.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fitter: Type[SED_code],
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        #top_hat_width: u.Quantity = 100.0 * u.AA,
    ):
        # TODO: Quick fix, should be in parent rather than the label
        self.SED_fitter = SED_fitter
        self.ext_src_corrs = ext_src_corrs
        self.ext_src_uplim = ext_src_uplim
        self.ref_wav = ref_wav
        #self.top_hat_width = top_hat_width
        super().__init__(aper_diam, SED_fitter.label)

    @property
    def name(self: Self) -> str:
        """Short string identifier for this property.

        Returns
        -------
        `str`
            `"MUV_{SED_fitter.label}"`, with a suffix appended describing
            any extended-source correction applied.
        """
        ext_src_label = funcs.get_ext_src_corr_label(
            ext_src_key = self.ext_src_corrs,
            ext_src_uplim = self.ext_src_uplim,
        )
        return f"MUV_{self.SED_fitter.label}{ext_src_label}"

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for this property.

        Returns
        -------
        `str`
            `"MUV ({SED_fitter.label})"`.
        """
        return f"MUV ({self.SED_fitter.label})"

    def _call_cat(
        self: Self,
        cat: Catalogue
    ) -> Catalogue:
        if self.ext_src_corrs is not None:
            # load extended source corrections
            cat.load_sextractor_ext_src_corrs()
        return super()._call_cat(cat)

    def _call_sed_result(
        self: Self,
        sed_result: SED_result,
    ) -> float:
        MUV = sed_result.SED.calc_MUV()
        ext_src_corr = funcs.get_ext_src_corr(
            sed_result.phot_rest,
            ext_src_key = self.ext_src_corrs,
            ext_src_uplim = self.ext_src_uplim,
            ref_wav = self.ref_wav,
        )
        MUV -= 2.5 * np.log10(ext_src_corr)
        return MUV


class Ext_Src_Property_Calculator(SED_Property_Calculator):
    """Applies an extended-source aperture correction to an existing SED-fit property.

    Rescales an already-computed property (and its stored PDF) by an
    extended-source correction factor, either taken from an individual
    band or from the band nearest a given rest-frame reference wavelength.

    Parameters
    ----------
    property_name : `str`
        Name of the pre-existing property (already stored on the SED fit
        result) to which the extended-source correction is applied.
    plot_name : `str`
        Human-readable label for plotting.
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used in the SED fit.
    SED_fit_label : `str` or `Type[SED_code]`
        The SED fitting code (or its label) whose results this calculator
        uses.
    ext_src_corrs : `str`, optional
        Extended-source correction to apply: either `"UV"` (uses the band
        nearest `ref_wav`) or an individual band name. If `None`, no
        correction is applied. Default is `"UV"`.
    ext_src_uplim : `int` or `float`, optional
        Upper limit imposed on the extended-source correction factor.
        Default is `10.0`.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength used to select the extended-source
        correction band when `ext_src_corrs == "UV"`. Default is
        `1_500.0 * astropy.units.AA`.

    Attributes
    ----------
    ext_src_corrs : `str` or `None`
        Extended-source correction setting.
    ext_src_uplim : `int`, `float`, or `None`
        Upper limit on the extended-source correction factor.
    ref_wav : `astropy.units.Quantity`
        Reference wavelength used for extended-source correction band
        selection.
    property_name : `str`
        Name of the property being corrected.
    """

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
            assert self.ext_src_corrs in ["UV"] + all_filt_names
        if self.ext_src_uplim is not None:
            assert isinstance(self.ext_src_uplim, (int, float))
            assert self.ext_src_uplim > 0.0
        assert u.get_physical_type(self.ref_wav) == "length"
        self.property_name = property_name
        self._plot_name = plot_name
        super().__init__(aper_diam, SED_fit_label)
    
    @property
    def name(self: Self) -> str:
        """Short string identifier for this property.

        Returns
        -------
        `str`
            The corrected property's name, with `"_extsrc_{ext_src_corrs}"`
            and (if an upper limit is set) `"<{ext_src_uplim:.0f}"` suffixes
            appended when an extended-source correction is applied.
        """
        ext_src_label = f"_extsrc_{self.ext_src_corrs}" \
            if self.ext_src_corrs is not None else ""
        ext_src_lim_label = f"<{self.ext_src_uplim:.0f}" if \
            self.ext_src_uplim is not None and \
            self.ext_src_corrs is not None else ""
        return self.property_name + ext_src_label + ext_src_lim_label

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for this property.

        Returns
        -------
        `str`
            The plot label supplied at construction.
        """
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
            ref_band = gal.aper_phot[self.aper_diam].filterset.filt_names[np.argmin(np.abs(band_wavs - self.ref_wav))]
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
        sed_result.properties[self.name] = new_pdf.median
        sed_result.property_errs[self.name] = new_pdf.errs
        sed_result.property_PDFs[self.name] = new_pdf
        # TODO: save raw data at this point
        return sed_result


class Custom_SED_Property_Extractor(Property_Extractor, SED_Property_Calculator):
    """Extracts an arbitrary named SED-fit property without recomputation.

    Parameters
    ----------
    name : `str`
        Short identifier of the SED-fit property to extract (e.g. an
        attribute name already stored on the SED fit result).
    plot_name : `str`
        Human-readable label for plotting.
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used in the SED fit.
    SED_fit_label : `str` or `Type[SED_code]`
        The SED fitting code (or its label) whose results this extracts
        from.
    """

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
        """Short string identifier for this extracted property.

        Returns
        -------
        `str`
            The property name supplied at construction.
        """
        return self._name

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for this extracted property.

        Returns
        -------
        `str`
            The plot label supplied at construction.
        """
        return self._plot_name


class Redshift_Extractor(Property_Extractor, SED_Property_Calculator):
    """Extracts the best-fit redshift, `z`, from the SED fit result.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used in the SED fit.
    SED_fit_label : `str` or `Type[SED_code]`
        The SED fitting code (or its label) whose redshift this extracts.
    """

    @property
    def name(self: Self) -> str:
        """Short string identifier for this property.

        Returns
        -------
        `str`
            Always `"z"`.
        """
        return "z"

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for this property.

        Returns
        -------
        `str`
            Always `"Redshift, z"`.
        """
        return "Redshift, z"


class Multiple_Property_Calculator(Property_Calculator_Base):
    """Abstract base class for calculators combining two or more sub-calculators.

    Parameters
    ----------
    calculators : `list` of `Property_Calculator_Base`
        The sub-calculators to combine.
    name : `str`, optional
        Short string identifier overriding any name derived from
        `calculators`. Default is `None`.
    plot_name : `str`, optional
        Human-readable label overriding any label derived from
        `calculators`. Default is `None`.

    Attributes
    ----------
    calculators : `list` of `Property_Calculator_Base`
        The sub-calculators being combined.
    """

    def __init__(
        self: Self,
        calculators: List[Type[Property_Calculator_Base]],
        name: Optional[str] = None,
        plot_name: Optional[str] = None,
    ) -> None:
        self._name = name
        self._plot_name = plot_name
        self.calculators = calculators

    @property
    @abstractmethod
    def full_name(self: Self) -> str:
        """Unique string identifier for the combined property.

        Returns
        -------
        `str`
            The fully-qualified combined property name.
        """
        pass

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """Short string identifier for the combined property.

        Returns
        -------
        `str`
            The combined property name, used as a dictionary/column key.
        """
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        """Human-readable label for the combined property.

        Returns
        -------
        `str`
            The plot label, potentially containing LaTeX math.
        """
        pass

    def __call__(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
        *args,
        **kwargs,
    ) -> Optional[Union[Type[Catalogue_Base], Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]]:
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
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
        try:
            return np.array([self._call_gal(gal) for gal in cat])
        except:
            unit = self._call_gal(cat.gals[0]).unit
            return np.array([self._call_gal(gal).value for gal in cat]) * unit
    
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


class Property_Multiplier(Multiple_Property_Calculator):
    """Computes the product of two or more sub-calculator properties.

    Parameters
    ----------
    calculators : `list` of `Property_Calculator_Base`
        The sub-calculators whose extracted values are multiplied together.
    name : `str`, optional
        Short string identifier overriding any name derived from
        `calculators`. Default is `None`.
    plot_name : `str`, optional
        Human-readable label overriding any label derived from
        `calculators`. Default is `None`.
    """

    def __init__(
        self: Self,
        calculators: List[Type[Property_Calculator_Base]],
        name: Optional[str] = None,
        plot_name: Optional[str] = None,
    ) -> None:
        super().__init__(calculators, name, plot_name)

    @property
    def full_name(self: Self) -> str:
        """Unique string identifier for the product property.

        Returns
        -------
        `str`
            The sub-calculator `full_name`s joined by `"_mult_"`, with any
            shared suffix (e.g. aperture diameter) moved to the end.
        """
        suffixes = [calculator.full_name.replace(calculator.name, "") for calculator in self.calculators]
        if all(suffix == suffixes[0] for suffix in suffixes):
            # add suffix to end of string
            return f"{'_mult_'.join([calculator.name for calculator in self.calculators])}{suffixes[0]}"
        else:
            return "_mult_".join([calculator.full_name for calculator in self.calculators])

    @property
    def name(self: Self) -> str:
        """Short string identifier for the product property.

        Returns
        -------
        `None`
            Not yet implemented; this property currently always returns
            `None`.
        """
        pass
        # if self._name is None:
        #     return "*".join([calculator.name for calculator in self.calculators])
        # else:
        #     return self._name

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for the product property.

        Returns
        -------
        `str`
            The `plot_name` supplied at construction, or the concatenation
            of the sub-calculators' `plot_name`s if none was supplied.
        """
        if self._plot_name is None:
            return "".join([calculator.plot_name for calculator in self.calculators])
        else:
            return self._plot_name

    def extract_vals(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the product of the sub-calculators' property values.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, `Photometry_rest`, `SED_obs`, or `Band_Cutout_Base`
            The object to extract and multiply the sub-calculator values
            for.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            Not currently implemented; this method always hits a
            `breakpoint()` before returning.
        """
        # calculate relevant values using each calculator
        vals = [calculator.extract_vals(object) for calculator in self.calculators]
        breakpoint()

    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the combined posterior PDF of the product property.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, `Photometry_rest`, `SED_obs`, or `Band_Cutout_Base`
            The object to extract and combine the sub-calculator PDFs for.

        Returns
        -------
        `PDF` or `list` of `PDF`
            Not currently implemented; this method always hits a
            `breakpoint()` before returning.
        """
        # calculate relevant PDFs using each calculator
        PDFs = [calculator.extract_PDFs(object) for calculator in self.calculators]
        breakpoint()

    #@abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        raise NotImplementedError


class Property_Divider(Multiple_Property_Calculator):
    """Computes the ratio of exactly two sub-calculator properties.

    Parameters
    ----------
    calculators : `list` of `Property_Calculator_Base`
        Exactly two sub-calculators; the first is divided by the second.
    name : `str`, optional
        Short string identifier overriding any name derived from
        `calculators`. Default is `None`.
    plot_name : `str`, optional
        Human-readable label overriding any label derived from
        `calculators`. Default is `None`.
    """

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
        """Unique string identifier for the ratio property.

        Returns
        -------
        `str`
            The sub-calculator `full_name`s joined by `"_div_"`, with any
            shared suffix (e.g. aperture diameter) moved to the end.
        """
        suffixes = [calculator.full_name.replace(calculator.name, "") for calculator in self.calculators]
        if all(suffix == suffixes[0] for suffix in suffixes):
            # add suffix to end of string
            return f"{'_div_'.join([calculator.name for calculator in self.calculators])}{suffixes[0]}"
        else:
            return "_div_".join([calculator.full_name for calculator in self.calculators])

    @property
    def name(self: Self) -> str:
        """Short string identifier for the ratio property.

        Returns
        -------
        `None`
            Not yet implemented; this property currently always returns
            `None`.
        """
        pass
        # if self._name is None:
        #     return "/".join([calculator.name for calculator in self.calculators])
        # else:
        #     return self._name

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for the ratio property.

        Returns
        -------
        `str`
            The `plot_name` supplied at construction, or the sub-
            calculators' `plot_name`s joined by `"/"` if none was supplied.
        """
        # TODO: Improve this so that units appear at the end of the string
        if self._plot_name is None:
            return "/".join([calculator.plot_name for calculator in self.calculators])
        else:
            return self._plot_name

    def extract_vals(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the ratio of the two sub-calculators' property values.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, `Photometry_rest`, `SED_obs`, or `Band_Cutout_Base`
            The object to extract and divide the sub-calculator values for.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            The first sub-calculator's value divided by the second's, with
            any `astropy.units.Dex` values converted to physical units
            first.
        """
        # calculate relevant values using each calculator
        vals_arr = [calculator.extract_vals(object) for calculator in self.calculators]
        # remove dex units
        vals_arr = [vals.physical if isinstance(vals, u.Dex) else vals for vals in vals_arr]
        return vals_arr[0] / vals_arr[1]

    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the combined posterior PDF of the ratio property.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, `Photometry_rest`, `SED_obs`, or `Band_Cutout_Base`
            The object to extract and divide the sub-calculator PDFs for.

        Returns
        -------
        `PDF`
            The PDF built from dividing the first sub-calculator's PDF
            input array by the second's, with any `astropy.units.Dex`
            input arrays converted to physical units first.
        """
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

# class Property_Adder(Multiple_Property_Calculator):

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

# class Property_Subtractor(Multiple_Property_Calculator):

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


class Re_kpc_Calculator(Multiple_Property_Calculator):
    """Converts a fitted effective radius from pixels to physical kpc.

    Combines a morphological effective-radius extractor with a redshift
    extractor to convert the fitted radius (in pixels) into a physical
    size, using the angular diameter distance at the fitted redshift.

    Parameters
    ----------
    re_extractor : `Custom_Morphology_Property_Extractor`
        Extractor for the fitted effective radius, in pixels.
    z_extractor : `Redshift_Extractor`
        Extractor for the redshift used to convert angular to physical
        size.
    """

    def __init__(
        self: Self,
        re_extractor: Custom_Morphology_Property_Extractor,
        z_extractor: Redshift_Extractor,
    ) -> None:
        super().__init__([re_extractor, z_extractor])

    @property
    def full_name(self: Self) -> str:
        """Unique string identifier for the kpc-converted radius property.

        Returns
        -------
        `str`
            `"{re_extractor.full_name}_kpc_z={z_label}"`.
        """
        return f"{self.calculators[0].full_name}_kpc_z={self.z_label}"

    @property
    def name(self: Self) -> str:
        """Short string identifier for the kpc-converted radius property.

        Returns
        -------
        `str`
            `"{re_extractor.name}_kpc"`.
        """
        return f"{self.calculators[0].name}_kpc" #_z={self.z_label}"

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for the kpc-converted radius property.

        Returns
        -------
        `str`
            The effective-radius extractor's `plot_name`, with `"pix"`
            replaced by `"kpc"`.
        """
        # TODO: Units should be pulled out into a separate property
        return self.calculators[0].plot_name.replace("pix", "kpc")

    @property
    def z_label(self: Self) -> str:
        """Label identifying the redshift extractor used for the conversion.

        Returns
        -------
        `str`
            The redshift extractor's `full_name` with its own `name`
            prefix removed.
        """
        return self.calculators[1].full_name.replace(f"{self.calculators[1].name}_", "")

    @property
    def pix_scale(self: Self) -> u.Quantity:
        """Pixel scale equivalency of the morphological fit's cutout.

        Returns
        -------
        `astropy.units.Quantity`
            An `astropy.units.pixel_scale` equivalency built from the
            effective-radius extractor's cutout pixel scale, usable to
            convert between pixels and angular units.
        """
        return u.pixel_scale(self.calculators[0].morph_fitter \
            .psf.cutout.band_data.pix_scale / u.pix)

    def _convert_to_kpc(
        self: Self,
        re_pix: u.Quantity,
        z: u.Quantity,
    ) -> u.Quantity:
        re_as = (re_pix).to(u.arcsec, self.pix_scale)
        return (re_as * astropy_cosmo.angular_diameter_distance(z)) \
            .to(u.kpc, u.dimensionless_angles())

    def extract_vals(
        self: Self,
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the effective radius converted to kpc.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Band_Cutout_Base`
            The object to extract the effective radius and redshift from.

        Returns
        -------
        `astropy.units.Quantity`
            The effective radius in kpc.
        """
        re_pix = self.calculators[0].extract_vals(object)
        z = self.calculators[1].extract_vals(object)
        return self._convert_to_kpc(re_pix, z)

    def extract_errs(
        self: Self,
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the lower/upper effective radius uncertainties, converted to kpc.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Band_Cutout_Base`
            The object to extract the effective radius PDFs and redshift
            from.

        Returns
        -------
        `astropy.units.Quantity`
            Array of shape `(N, 2)` (or `(2,)` for a single object) giving
            the lower and upper effective radius uncertainties in kpc.
            Entries are `numpy.nan` where no PDF is available.
        """
        re_pix_errs = np.array([pdf.errs if pdf is not None else [np.nan, np.nan] for pdf in self.calculators[0].extract_PDFs(object)]) * u.pix
        z = self.calculators[1].extract_vals(object)
        re_kpc_errs = self._convert_to_kpc(re_pix_errs, z[:, np.newaxis])
        return re_kpc_errs.T

    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the posterior PDF of the effective radius, converted to kpc.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Band_Cutout_Base`
            The object to extract the effective radius PDFs and redshift
            from.

        Returns
        -------
        `list` of `PDF`
            One kpc-converted effective-radius PDF per galaxy in `object`.
        """
        re_pix_pdfs = self.calculators[0].extract_PDFs(object)
        z = self.calculators[1].extract_vals(object)
        return [PDF.from_1D_arr(self.name, self._convert_to_kpc( \
            pdfs.input_arr, z_)) for pdfs, z_ in zip(re_pix_pdfs, z)]
    
    #@abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        [calculator(gal) for calculator in self.calculators]
        return self.extract_vals(gal)


class Surface_Density_Calculator(Multiple_Property_Calculator):
    """Computes a surface density, Sigma = property / (2 pi Re_kpc^2).

    Divides a given property (e.g. stellar mass, SFR) by the area implied
    by a kpc-converted effective radius, to give a surface density.

    Parameters
    ----------
    property_calculator : `Property_Calculator_Base`
        Calculator for the property (e.g. mass or SFR) to convert to a
        surface density.
    re_kpc_calculator : `Re_kpc_Calculator`
        Calculator for the effective radius in kpc, used to compute the
        implied area.
    """

    def __init__(
        self: Self,
        property_calculator: Type[Property_Calculator_Base],
        re_kpc_calculator: Re_kpc_Calculator,
    ) -> None:
        super().__init__([property_calculator, re_kpc_calculator])

    @classmethod
    def from_re_pix(
        cls,
        property_calculator: Type[Property_Calculator_Base],
        re_extractor: Custom_Morphology_Property_Extractor,
        z_extractor: Redshift_Extractor,
    ) -> Self:
        """Construct a `Surface_Density_Calculator` from a pixel-space radius extractor.

        Builds the intermediate `Re_kpc_Calculator` from `re_extractor` and
        `z_extractor` before constructing the surface density calculator.

        Parameters
        ----------
        property_calculator : `Property_Calculator_Base`
            Calculator for the property to convert to a surface density.
        re_extractor : `Custom_Morphology_Property_Extractor`
            Extractor for the fitted effective radius, in pixels.
        z_extractor : `Redshift_Extractor`
            Extractor for the redshift used to convert angular to physical
            size.

        Returns
        -------
        `Surface_Density_Calculator`
            The constructed surface density calculator.
        """
        re_kpc_calculator = Re_kpc_Calculator(re_extractor, z_extractor)
        return cls(property_calculator, re_kpc_calculator)

    @property
    def full_name(self: Self) -> str:
        """Unique string identifier for the surface density property.

        Returns
        -------
        `str`
            `"Sigma_{property_calculator.name}_{re_kpc_calculator.name}_{suffix}"`
            if the property calculator and radius calculator share a common
            suffix (e.g. SED-fit label), otherwise
            `"Sigma_{property_calculator.full_name}_{re_kpc_calculator.full_name}"`.
        """
        suffixes = [
            self.calculators[0].full_name.replace(f"{self.calculators[0].name}_", ""),
            self.calculators[1].z_label,
        ]
        if all(suffix == suffixes[0] for suffix in suffixes):
            # add suffix to end of string
            return f"Sigma_{self.calculators[0].name}_{self.calculators[1].name}_{suffixes[0]}"
        else:
            return f"Sigma_{self.calculators[0].full_name}_{self.calculators[1].full_name}"

    @property
    def name(self: Self) -> str:
        """Short string identifier for the surface density property.

        Returns
        -------
        `str`
            `"Sigma_{property_calculator.name}_{re_kpc_calculator.name}"`.
        """
        return f"Sigma_{self.calculators[0].name}_{self.calculators[1].name}"

    @property
    def plot_name(self: Self) -> str:
        """Human-readable label for the surface density property.

        Returns
        -------
        `str`
            LaTeX-formatted `r"$\\Sigma_{...}$"` label built from the
            property calculator's `plot_name`.
        """
        # TODO: Units should be pulled out into a separate property
        return rf"$\Sigma_{{{self.calculators[0].plot_name.replace('$', '')}}}$"
    
    def _convert_to_surface_density(
        self: Self,
        property_val: Union[u.Quantity, u.Dex],
        re_kpc: u.Quantity,
    ) -> u.Quantity:
        if isinstance(property_val, u.Dex):
            property_val = property_val.physical
        return property_val / (2 * np.pi * re_kpc.to(u.kpc) ** 2)

    def extract_vals(
        self: Self,
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]],
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the surface density value(s).

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Band_Cutout_Base`
            The object to extract the property value and effective radius
            from.

        Returns
        -------
        `astropy.units.Quantity`
            The property value divided by `2 * pi * Re_kpc ** 2`.
        """
        property_vals = self.calculators[0].extract_vals(object)
        re_kpc = self.calculators[1].extract_vals(object)
        return self._convert_to_surface_density(property_vals, re_kpc)

    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the posterior PDF of the surface density.

        Parameters
        ----------
        object : `Catalogue`, `Galaxy`, or `Band_Cutout_Base`
            The object to extract the property PDFs and effective radius
            from.

        Returns
        -------
        `list` of `PDF`
            One surface-density PDF per galaxy in `object`, built by
            dividing each property PDF's input array by
            `2 * pi * Re_kpc ** 2`.
        """
        property_pdfs = self.calculators[0].extract_PDFs(object)
        re_kpc = self.calculators[1].extract_vals(object)
        return [PDF.from_1D_arr(self.name, self._convert_to_surface_density( \
            pdfs.input_arr, re_kpc_)) for pdfs, re_kpc_ in zip(property_pdfs, re_kpc)]
    
    #@abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        raise NotImplementedError
    
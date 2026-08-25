"""Astronomical facility and instrument definitions.

Defines Facility class representing observing facilities (HST,
JWST, Spitzer, etc.)
and Instrument subclasses for specific instruments (NIRCam, MIRI, ACS_WFC,
etc.)
with their filter configurations and metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from . import Band_Data, Band_Data_Base, PSF_Base, PSF_Cutout

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import astropy.units as u
import h5py
import numpy as np
from astropy.io import ascii
from astropy.table import Table

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import config, galfind_logger
from ..utils import useful_funcs_austind as funcs
from ..utils.exceptions import (
    InvalidOptionError,
    MissingFileError,
    MissingKeyError,
)
from .PSF import PSF_Cutout


class Facility(ABC):
    """Abstract base class representing an observing facility (telescope).

    Stores the name of the facility used for SVO Filter Profile Service
    queries and provides consistent identity, string, and (deep)copy
    semantics. Subclasses are combined with
    `useful_funcs_austind.Singleton` so that only one instance of each
    facility ever exists.

    Attributes
    ----------
    SVO_name : `str`
        Name used to query this facility in the SVO Filter Profile
        Service. Defaults to the subclass name if not otherwise set.
    """

    def __init__(self) -> None:
        if not hasattr(self, "SVO_name"):
            self.SVO_name = self.__class__.__name__

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: Type[Self]) -> bool:
        if isinstance(other, Facility):
            return self.__class__.__name__ == other.__class__.__name__
        else:
            return False

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result


class HST(Facility, funcs.Singleton):
    """Singleton `Facility` representing the Hubble Space Telescope."""

    pass


class JWST(Facility, funcs.Singleton):
    """Singleton `Facility` representing the James Webb Space Telescope."""

    pass


class Paranal(Facility, funcs.Singleton):
    """Singleton `Facility` representing ESO's Paranal Observatory (VISTA)."""

    pass


class Spitzer(Facility, funcs.Singleton):
    """Singleton `Facility` representing the Spitzer Space Telescope."""

    pass


class Euclid(Facility, funcs.Singleton):
    """Singleton `Facility` representing the Euclid space telescope."""

    pass


class CFHT(Facility, funcs.Singleton):
    """Singleton `Facility` representing the Canada-France-Hawaii Telescope."""

    pass


class Subaru(Facility, funcs.Singleton):
    """Singleton `Facility` representing the Subaru Telescope."""

    pass


class Instrument(ABC):
    """Abstract base class representing an imaging instrument on a facility.

    Stores the filter names available on the instrument together with
    facility information and the parameters used for astrometric
    alignment. Concrete subclasses (e.g. `NIRCam`, `ACS_WFC`) must
    implement the `ZP_keys` property, `calc_ZP`, and `make_model_psf`.

    Parameters
    ----------
    facility : `Facility`
        Facility (telescope) that hosts this instrument.
    filt_names : `list` of `str`
        Names of the filters/bands available on this instrument.
    align_params : `dict`, optional
        Parameters used for astrometric alignment (e.g. ``searchrad``,
        ``separation``, ``tolerance``, ``max_sep``). Default is `{}`.

    Attributes
    ----------
    facility : `Facility`
        Facility hosting this instrument.
    filt_names : `list` of `str`
        Names of the filters available on this instrument.
    align_params : `dict`
        Astrometric alignment parameters.
    SVO_name : `str`
        Name used to query this instrument in the SVO Filter Profile
        Service. Defaults to the subclass name if not otherwise set.
    """

    def __init__(
        self: Type[Self],
        facility: Facility,
        filt_names: List[str],
        align_params: Dict[str, Any] = {},
    ) -> None:
        self.facility = facility
        self.filt_names = filt_names
        self.align_params = align_params

        if not hasattr(self, "SVO_name"):
            self.SVO_name = self.__class__.__name__

    def __str__(self) -> str:
        # print filter_names?
        output_str = funcs.line_sep
        output_str += (
            f"{self.facility.__class__.__name__}/{self.__class__.__name__}\n"
        )
        if len(self.facility.__dict__) > 0 or len(self.__dict__) > 0:
            output_str += funcs.line_sep
        if len(self.__dict__) > 0:
            for key, value in self.facility.__dict__.items():
                output_str += f"{key}: {value}\n"
            output_str += funcs.band_sep
        if len(self.facility.__dict__) > 0:
            output_str += "FACILITY:\n"
            output_str += funcs.band_sep
            for key, value in self.facility.__dict__.items():
                output_str += f"{key}: {value}\n"
        output_str += funcs.line_sep
        return output_str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: Type[Self]) -> bool:
        if isinstance(other, Instrument):
            return (
                self.facility == other.facility
                and self.__class__.__name__ == other.__class__.__name__
            )
        else:
            return False

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    @property
    @abstractmethod
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords required by `calc_ZP`.

        Subclasses must override this to return the header keywords
        (e.g. ``'PHOTFLAM'``, ``'ZEROPNT'``) needed to compute the
        zero-point, or an empty list if the zero-point does not depend
        on header keywords.
        """
        pass

    @abstractmethod
    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the photometric zero-point for the given band data.

        Subclasses must implement this to return the AB magnitude
        zero-point appropriate for the instrument and band, derived
        either from FITS header keywords or from a fixed formula.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data object providing access to the image and header
            needed to compute the zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band.
        """
        pass

    def calc_pix_scale(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the pixel scale for the given band data.

        Not implemented in the base class.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data object for which to determine the pixel scale.

        Returns
        -------
        `None`
            This base implementation performs no calculation.
        """
        pass

    def make_psf(
        self: Self,
        band_data: Band_Data,
        method: str = "default",
        size: u.Quantity = 0.96 * u.arcsec,
    ) -> Type[PSF_Base]:
        """Construct a PSF for the given band data using the requested method.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the PSF.
        method : `str`, optional
            PSF construction method, one of ``'default'``, ``'empirical'``,
            or ``'EPOCHS'``. Default is `'default'`.
        size : `astropy.units.Quantity`, optional
            Angular size of the PSF cutout. Default is `0.96 * u.arcsec`.

        Returns
        -------
        `PSF_Base`
            The constructed PSF. For ``'default'`` this delegates to
            `make_model_psf`; for ``'empirical'`` this loads/derives an
            empirical PSF via `PSF_Cutout.from_empirical_psf`; for
            ``'EPOCHS'`` this uses a model PSF for `ACS_WFC` instruments
            or loads a pre-computed EPOCHS PSF from disk otherwise.

        Raises
        ------
        InvalidOptionError
            If `method` is not one of ``'default'``, ``'empirical'``, or
            ``'EPOCHS'``.
        """
        method_types = ["default", "empirical", "EPOCHS"]
        if method not in method_types:
            raise InvalidOptionError(
                f"method={method!r} not in {method_types}."
            )
        if method == "default":
            return self.make_model_psf(
                band_data,
                size=size,
            )
        elif method == "empirical":
            return PSF_Cutout.from_empirical_psf(band_data)
        else:  # method == "EPOCHS":
            if isinstance(self, ACS_WFC):
                return self.make_model_psf(
                    band_data,
                    size=size,
                )
            else:
                base_path = f"{config['PSF']['PSF_WORK_DIR']}/EPOCHS_PSFs"
                epochs_path = (
                    f"{base_path}/{band_data.filt_name}/"
                    f"PSF_Resample_03_{band_data.filt_name}.fits"
                )
                return PSF_Cutout.from_fits(
                    epochs_path,
                    band_data,
                    size=size,
                    origin="webbpsf",
                )

    @abstractmethod
    def make_model_psf(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 0.96 * u.arcsec,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for the given band data.

        Subclasses must implement this to return a model PSF (e.g.
        generated from STPSF/WebbPSF, or derived from precomputed
        encircled energy curves), or raise `NotImplementedError` if
        model PSF construction is not supported for the instrument.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.
        size : `astropy.units.Quantity`, optional
            Angular size of the PSF cutout. Default is `0.96 * u.arcsec`.

        Returns
        -------
        `PSF_Base`
            The constructed model PSF object.
        """
        pass

    # def make_empirical_psf(
    #     self: Self,
    #     band_data: Band_Data,
    # ) -> Type[PSF_Base]:
    #     raise NotImplementedError(
    #         f"Empirical PSF construction not yet implemented for "
    #         f"{repr(self)}!"
    #     )

    def get_psf_norm(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 4.0 * u.arcsec,
    ) -> Optional[Tuple[float, float]]:
        """Get the encircled energy and field of view for PSF normalization.

        Base implementation always returns `None`; subclasses with
        encircled-energy calibration data available (e.g. `NIRCam`,
        `ACS_WFC`) override this.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to determine the PSF normalization.
        size : `astropy.units.Quantity`, optional
            Angular aperture used to determine the encircled energy.
            Default is `4.0 * u.arcsec`.

        Returns
        -------
        `None`
            This base implementation performs no calculation.
        """
        return None

    def get_psf_norm_path(
        self: Self,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Get the path to the PSF normalization (encircled energy) file.

        Base implementation always returns `None`; subclasses with
        calibration data available override this to return a file path.

        Parameters
        ----------
        **kwargs : `dict`
            Additional keyword arguments, unused in the base
            implementation.

        Returns
        -------
        `None`
            This base implementation performs no lookup.
        """
        return None

    @staticmethod
    def get_psf_dir(band_data: Band_Data) -> str:
        """Get the working directory for PSF-related files for a band.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data used to determine the instrument, version, survey,
            and filter name making up the directory path.

        Returns
        -------
        `str`
            Path to the PSF working directory for this band.
        """
        return (
            f"{config['PSF']['PSF_WORK_DIR']}/{band_data.filt.instrument.__class__.__name__}"
            + f"/{band_data.version}/{band_data.survey}/{band_data.filt_name}"
        )

    def get_eec_path(
        self: Self,
        band_data: Band_Data,
    ) -> str:
        """Get the path to the encircled energy curve (EEC) file for a band.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to determine the EEC file path.

        Returns
        -------
        `str`
            Path to the HDF5 file storing the encircled energy curve.
        """
        eec_name = f"EEC_{band_data.filt.filt_name}.h5"
        eec_path = f"{self.get_psf_dir(band_data)}/{eec_name}"
        return eec_path


class NIRCam(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing JWST/NIRCam.

    Defines the full set of NIRCam filters and the astrometric alignment
    parameters used for NIRCam mosaics.

    Attributes
    ----------
    facility : `JWST`
        The JWST facility instance hosting NIRCam.
    filt_names : `list` of `str`
        Names of all NIRCam filters.
    align_params : `dict`
        Astrometric alignment parameters for NIRCam.
    """

    def __init__(self) -> None:
        NIRCam_filt_names = [
            "F070W",
            "F090W",
            "F115W",
            "F140M",
            "F150W",
            "F162M",
            "F164N",
            "F150W2",
            "F182M",
            "F187N",
            "F200W",
            "F210M",
            "F212N",
            "F250M",
            "F277W",
            "F300M",
            "F323N",
            "F322W2",
            "F335M",
            "F356W",
            "F360M",
            "F405N",
            "F410M",
            "F430M",
            "F444W",
            "F460M",
            "F466N",
            "F470N",
            "F480M",
        ]
        align_params = {
            "searchrad": 40,
            "separation": 0.09,
            "tolerance": 10.0,
            "max_sep": 1000,
        }
        super().__init__(JWST(), NIRCam_filt_names, align_params)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords required to calculate
        the zero-point (empty; NIRCam uses a fixed pixel-scale-based
        formula)."""
        return []

    def calc_ZP(self: Self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point assuming MJy/sr flux units.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the pixel scale used in the zero-point
            calculation.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band.
        """
        # assume flux units of MJy/sr and calculate corresponding ZP
        ZP = -2.5 * np.log10(
            (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
        ) + u.Jy.to(u.ABmag)
        return ZP

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 0.96 * u.arcsec,
    ) -> PSF_Cutout:
        """Construct a model PSF for NIRCam using STPSF (formerly WebbPSF).

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.
        size : `astropy.units.Quantity`, optional
            Angular size of the PSF cutout. Currently unused (kept for
            interface compatibility with `Instrument.make_model_psf`).
            Default is `0.96 * u.arcsec`.

        Returns
        -------
        `PSF_Cutout`
            The constructed model PSF cutout.
        """
        from . import PSF_Cutout

        # TODO: create PSF_Cutout from WebbPSF instead of loading from file
        # psf_path = f"{psf_dir}/NIRCam/{band_data.filt.filt_name}.fits"
        psf = PSF_Cutout.from_stpsf(
            band_data,
            # psf_path,
            # band_data.filt,
            # size = size,
        )
        return psf

    def get_psf_norm(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 4.0 * u.arcsec,
    ) -> Union[None, Tuple[float, float]]:
        """Get the encircled energy and field of view for PSF normalization.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to determine the PSF normalization.
        size : `astropy.units.Quantity`, optional
            Angular aperture (diameter) used to look up the tabulated
            encircled energy. Default is `4.0 * u.arcsec`.

        Returns
        -------
        `tuple` of (`float`, `float`) or `None`
            The encircled energy fraction and the aperture diameter
            (arcsec) it corresponds to, or `None` if no normalization
            file could be found for this filter.

        Raises
        ------
        MissingFileError
            If the normalization file returned by `get_psf_norm_path`
            does not exist.
        """
        norm_path = self.get_psf_norm_path(band_data=band_data)
        if norm_path is None:
            return None
        else:
            if not Path(norm_path).is_file():
                raise MissingFileError(
                    f"PSF normalization file norm_path={norm_path!r} "
                    "not found!"
                )
            energy_table = ascii.read(norm_path)
            row = np.argmin(
                abs(
                    size.to(u.arcsec).value / 2.0 - energy_table["aper_radius"]
                )
            )
            encircled = energy_table[row][band_data.filt.filt_name]
            norm_fov = energy_table["aper_radius"][row] * 2
            galfind_logger.debug(
                f"Normalizing PSF within {norm_fov} FOV to {encircled}"
            )
            return encircled, norm_fov

    def get_psf_norm_path(
        self: Self,
        band_data: Band_Data,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Get the path to the encircled energy calibration file for a band.

        Selects between the short-wavelength (SW) and long-wavelength (LW)
        NIRCam encircled energy tables depending on which one contains the
        requested filter.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data whose filter determines which table is returned.
        **kwargs : `dict`
            Additional keyword arguments, unused.

        Returns
        -------
        `str` or `None`
            Path to the SW or LW encircled energy text file, or `None`
            if the filter is not found in either table.
        """
        base_psf_dir = f"{config['PSF']['PSF_DIR']}/{self.__class__.__name__}"
        ee_dir = f"{base_psf_dir}/encircled_energy"
        SW_norm_path = f"{ee_dir}/Encircled_Energy_SW_ETCv2.txt"
        LW_norm_path = f"{ee_dir}/Encircled_Energy_LW_ETCv2.txt"
        filters = {
            "SW": Table.read(SW_norm_path, format="ascii").colnames[1:],
            "LW": Table.read(LW_norm_path, format="ascii").colnames[1:],
        }
        if band_data.filt.filt_name in filters["SW"]:
            return SW_norm_path
        elif band_data.filt.filt_name in filters["LW"]:
            return LW_norm_path
        else:
            galfind_logger.warning(
                f"Filter {band_data.filt.filt_name} not found in any "
                "PSF normalization table!"
            )
            return None


class MIRI(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing JWST/MIRI.

    Defines the full set of MIRI imaging filters.

    Attributes
    ----------
    facility : `JWST`
        The JWST facility instance hosting MIRI.
    filt_names : `list` of `str`
        Names of all MIRI filters.
    """

    def __init__(self) -> None:
        MIRI_filt_names = [
            "F560W",
            "F770W",
            "F1000W",
            "F1065C",
            "F1140C",
            "F1130W",
            "F1280W",
            "F1500W",
            "F1550C",
            "F1800W",
            "F2100W",
            "F2300C",
            "F2550W",
        ]
        super().__init__(JWST(), MIRI_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords required to calculate
        the zero-point (empty; MIRI uses a fixed pixel-scale-based
        formula)."""
        return []

    def calc_ZP(self: Self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point assuming MJy/sr flux units.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the pixel scale used in the zero-point
            calculation.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band.
        """
        # assume flux units of MJy/sr and calculate corresponding ZP
        ZP = -2.5 * np.log10(
            (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
        ) + u.Jy.to(u.ABmag)
        return ZP

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 0.96 * u.arcsec,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for MIRI.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.
        size : `astropy.units.Quantity`, optional
            Angular size of the PSF cutout. Default is `0.96 * u.arcsec`.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for MIRI.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for MIRI!"
        )


class ACS_SBC(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        ACS_SBC_band_names = [
            "F122M",
            "F115LP",
            "F125LP",
            "PR130L",
            "PR110L",
            "F140LP",
            "F150LP",
            "F165LP",
        ]
        self.SVO_name = "ACS"
        super().__init__(HST(), ACS_SBC_band_names)

    @property
    def ZP_keys(self) -> List[str]:
        """FITS header keywords required to calculate zero-point:
        PHOTFLAM, PHOTPLAM, or ZEROPNT."""
        return ["PHOTFLAM", "PHOTPLAM", "ZEROPNT"]

    def calc_ZP(self: Self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        im_header = band_data.load_im()[1]
        if "PHOTFLAM" in im_header and "PHOTPLAM" in im_header:
            ZP = (
                -2.5 * np.log10(im_header["PHOTFLAM"])
                - 21.1
                - 5.0 * np.log10(im_header["PHOTPLAM"])
                + 18.6921
            )
        elif "ZEROPNT" in im_header:
            ZP = im_header["ZEROPNT"]
            galfind_logger.warning(
                f"Using ZEROPNT from header for {band_data.filt_name} "
                "instead of PHOTFLAM and PHOTPLAM! Potential ST mags "
                "used here!"
            )
        else:
            raise MissingKeyError(
                f"{self.__class__.__name__} data for "
                f"band_data.filt_name={band_data.filt_name!r} must "
                "contain either 'ZEROPNT' or 'PHOTFLAM' and 'PHOTPLAM' "
                f"in its header to calculate its ZP; available keys="
                f"{list(im_header.keys())!r}."
            )  # or 'BUNIT'=MJy/sr
        return ZP

    def make_model_psf(
        self: Self, band_data: Band_Data, size: u.Quantity = 0.96 * u.arcsec
    ) -> Type[PSF_Base]:
        """Model PSF construction not supported for HST/ACS-SBC."""
        raise NotImplementedError(
            "Model PSF generation is not supported for HST/ACS-SBC"
        )

    def make_empirical_PSF(self: Self, band_data: Band_Data) -> Type[PSF_Base]:
        pass


class ACS_WFC(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing HST/ACS-WFC.

    Defines the full set of ACS/WFC filters and the astrometric alignment
    parameters used for ACS/WFC mosaics. Uses the SVO instrument name
    ``'ACS'``.

    Attributes
    ----------
    facility : `HST`
        The HST facility instance hosting ACS/WFC.
    filt_names : `list` of `str`
        Names of all ACS/WFC filters.
    align_params : `dict`
        Astrometric alignment parameters for ACS/WFC.
    SVO_name : `str`
        SVO instrument name (``'ACS'``).
    """

    def __init__(self) -> None:
        ACS_WFC_filt_names = [
            "FR388N",
            "FR423N",
            "F435W",
            "FR459M",
            "FR462N",
            "F475W",
            "F502N",
            "FR505N",
            "F555W",
            # "FR551N",
            "F550M",
            "F606W",
            "FR601N",
            "F625W",
            "FR647M",
            "FR656N",
            "F658N",
            "F660N",
            "FR716N",
            # "POL_UV",
            "G800L",
            # "POL_V",
            "F775W",
            "FR782N",
            "F814W",
            "FR853N",
            "F892N",
            "F850LP",
            "FR914M",
            "FR931N",
            "FR1016N",
        ]
        align_params = {
            "searchrad": 30,
            "separation": 0.03,
            "tolerance": 8.0,
            "max_sep": 100,
        }
        self.SVO_name = "ACS"
        super().__init__(HST(), ACS_WFC_filt_names, align_params)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords used to calculate the
        zero-point ('PHOTFLAM', 'PHOTPLAM', 'ZEROPNT')."""
        return ["PHOTFLAM", "PHOTPLAM", "ZEROPNT"]  # or 'BUNIT'=MJy/sr

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from the image header.

        Uses 'PHOTFLAM' and 'PHOTPLAM' if both are present, otherwise
        falls back to 'ZEROPNT'.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the image header used to compute the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band.

        Raises
        ------
        MissingKeyError
            If neither ('PHOTFLAM' and 'PHOTPLAM') nor 'ZEROPNT' are
            present in the image header.
        """
        im_header = band_data.load_im()[1]
        if "PHOTFLAM" in im_header and "PHOTPLAM" in im_header:
            ZP = (
                -2.5 * np.log10(im_header["PHOTFLAM"])
                - 21.1
                - 5.0 * np.log10(im_header["PHOTPLAM"])
                + 18.6921
            )
        elif "ZEROPNT" in im_header:
            ZP = im_header["ZEROPNT"]
            galfind_logger.debug(
                f"Using ZEROPNT from header for {band_data.filt_name} "
                "instead of PHOTFLAM and PHOTPLAM! Potential ST mags "
                "used here!"
            )
        # DO NOT UNCOMMENT! REGULARLY CAUSES ZP ERRORS
        # elif "BUNIT" in im_header:
        #     unit = im_header["BUNIT"].replace(" ", "")
        #     assert unit == "MJy/sr"
        #     ZP = -2.5 * np.log10(
        #         (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
        #     ) + u.Jy.to(u.ABmag)
        else:
            raise MissingKeyError(
                f"ACS_WFC data for band_data.filt_name="
                f"{band_data.filt_name!r} must contain either 'PHOTFLAM' "
                "and 'PHOTPLAM' or 'ZEROPNT' in its header to calculate "
                f"its ZP; available keys={list(im_header.keys())!r}."
            )  # or 'BUNIT'=MJy/sr
        return ZP

    def get_psf_norm_path(
        self: Self,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Get the path to the ACS/WFC encircled energy calibration file.

        Parameters
        ----------
        **kwargs : `dict`
            Additional keyword arguments, unused.

        Returns
        -------
        `str`
            Path to the ACS/WFC encircled energy text file.
        """
        psf_dir = config["PSF"]["PSF_DIR"]
        class_name = self.__class__.__name__
        return f"{psf_dir}/{class_name}/encircled_energy/ACS_WFC_EE.txt"

    def get_psf_norm(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 4.0 * u.arcsec,
    ) -> Tuple[float, float]:
        """Get the encircled energy and field of view for PSF normalization.

        Loads (creating if necessary, via `_make_eec`) the encircled
        energy curve for `band_data` and looks up the tabulated value
        nearest to the requested aperture size.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to determine the PSF normalization.
        size : `astropy.units.Quantity`, optional
            Angular aperture (diameter) used to look up the tabulated
            encircled energy. Default is `4.0 * u.arcsec`.

        Returns
        -------
        `tuple` of (`float`, `float`)
            The encircled energy fraction and the aperture diameter
            (arcsec) it corresponds to.
        """
        eec_path = self.get_eec_path(band_data)
        self._make_eec(band_data)
        with h5py.File(eec_path, "r") as f:
            radii = f["radii"][:]
            eec = f["eec"][:]
        row = np.argmin(abs(size.to(u.arcsec).value / 2.0 - radii))
        encircled = eec[row]
        norm_fov = radii[row] * 2 * u.arcsec
        galfind_logger.debug(
            f"Normalizing {repr(self)} within {norm_fov=} to {encircled}"
        )
        return encircled, norm_fov

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
        **kwargs: Dict[str, Any],
    ) -> PSF_Base:
        """Construct a model PSF for ACS/WFC from its encircled energy curve.

        Ensures the encircled energy curve for `band_data` exists (via
        `_make_eec`), then builds a `PSF_Base` from it.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.
        **kwargs : `dict`
            Additional keyword arguments forwarded to the `PSF_Base`
            constructor.

        Returns
        -------
        `PSF_Base`
            The constructed model PSF object.
        """
        from . import PSF_Base

        self._make_eec(band_data)
        return PSF_Base(
            self.get_eec_path(band_data),
            name=f"{band_data.filt_name}_model",
            **kwargs,
        )

    def _make_eec(
        self: Self,
        band_data: Band_Data,
    ) -> None:
        """Create the encircled energy curve (EEC) HDF5 file for a band,
        if it does not already exist.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to create the EEC file.

        Raises
        ------
        MissingFileError
            If the encircled energy text file returned by
            `get_psf_norm_path` does not exist.
        InvalidOptionError
            If the encircled energy text file does not contain exactly
            one row for `band_data.filt.filt_name`.
        """
        eec_path = self.get_eec_path(band_data)
        if not Path(eec_path).is_file():
            galfind_logger.info(
                f"Creating EEC data for {repr(band_data.filt)} at {eec_path}"
            )
            txt_filepath = self.get_psf_norm_path(band_data=band_data)
            if not Path(txt_filepath).is_file():
                raise MissingFileError(
                    f"EE data for {self.__class__.__name__} not found in "
                    f"txt_filepath={txt_filepath!r}."
                )
            eec_tab = Table.read(txt_filepath, format="ascii.commented_header")
            eec_tab = eec_tab[eec_tab["band"] == band_data.filt.filt_name]
            if len(eec_tab) != 1:
                raise InvalidOptionError(
                    f"EE data for {repr(band_data.filt)} not found in "
                    f"txt_filepath={txt_filepath!r}; matched "
                    f"{len(eec_tab)} rows, expected exactly 1."
                )
            radii = np.zeros(len(eec_tab.colnames) - 1)  # * u.arcsec
            eec = np.zeros(len(eec_tab.colnames) - 1)
            for i, col in enumerate(eec_tab.colnames):
                if col != "band":
                    radii[i - 1] = float(col) * 0.05  # * u.arcsec
                    eec[i - 1] = eec_tab[col]
            radii = radii.astype(np.float32)
            eec = eec.astype(np.float32)
            # save eec data to h5 file
            funcs.make_dirs(eec_path)
            with h5py.File(eec_path, "w") as f:
                f.create_dataset("radii", data=radii, compression="gzip")
                f.create_dataset("eec", data=eec, compression="gzip")
                galfind_logger.info(
                    f"Saved EEC data for {repr(band_data.filt)} to {eec_path}"
                )
            funcs.change_file_permissions(eec_path)
        else:
            galfind_logger.debug(
                f"EEC data for {repr(band_data.filt)} already exists at "
                f"{eec_path}!"
            )


class WFC3_IR(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing HST/WFC3-IR.

    Defines the full set of WFC3-IR filters. Uses the SVO instrument name
    ``'WFC3'``.

    Attributes
    ----------
    facility : `HST`
        The HST facility instance hosting WFC3-IR.
    filt_names : `list` of `str`
        Names of all WFC3-IR filters.
    SVO_name : `str`
        SVO instrument name (``'WFC3'``).
    """

    def __init__(self) -> None:
        WFC3_IR_filt_names = [
            "F098M",
            "G102",
            "F105W",
            "F110W",
            "F125W",
            "F126N",
            "F127M",
            "F128N",
            "F130N",
            "F132N",
            "F139M",
            "F140W",
            "G141",
            "F153M",
            "F160W",
            "F164N",
            "F167N",
        ]
        self.SVO_name = "WFC3"
        super().__init__(HST(), WFC3_IR_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords required to calculate
        the zero-point (empty; WFC3-IR uses fixed per-filter
        zero-points)."""
        return []

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from a fixed
        per-filter lookup table.

        Uses tabulated AB zero-points from Appendix A of WFC3 ISR
        2020-10, valid for the ``F098M``, ``F105W``, ``F110W``,
        ``F125W``, ``F140W``, and ``F160W`` filters.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the filter used to look up the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band.

        Raises
        ------
        KeyError
            If `band_data.filt.filt_name` is not one of the tabulated
            WFC3-IR filters.
        """
        # Taken from Appendix A of
        # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2020/WFC3-ISR-2020-10.pdf
        wfc3ir_zps = {
            "F098M": 25.661,
            "F105W": 26.2637,
            "F110W": 26.8185,
            "F125W": 26.231,
            "F140W": 26.4502,
            "F160W": 25.9362,
        }
        return wfc3ir_zps[band_data.filt.filt_name]

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for WFC3-IR.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for WFC3-IR.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for WFC3-IR!"
        )


class VISTA(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing VISTA/VIRCam.

    Defines the full set of VISTA filters. Uses the SVO instrument name
    ``'VIRCam'``.

    Attributes
    ----------
    facility : `Paranal`
        The Paranal facility instance hosting VISTA.
    filt_names : `list` of `str`
        Names of all VISTA filters.
    SVO_name : `str`
        SVO instrument name (``'VIRCam'``).
    """

    def __init__(self) -> None:
        VISTA_filt_names = [
            "Z_filter",
            "Z",
            "NB980_filter",
            "NB980",
            "NB990_filter",
            "NB990",
            "Y_filter",
            "Y",
            "NB118_filter",
            "NB118",
            "J",
            "J_filter",
            "H",
            "H_filter",
            "Ks_filter",
            "Ks",
        ]
        self.SVO_name = "VIRCam"
        super().__init__(Paranal(), VISTA_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords used to calculate the
        zero-point (``'PHOTZP'``)."""
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from the image header.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the image header used to look up the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band, taken directly
            from the ``'PHOTZP'`` header keyword.
        """
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for VISTA.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for VISTA.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for VISTA!"
        )


class MegaCam(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing CFHT/MegaCam.

    Defines the full set of MegaCam filters. Uses the SVO instrument
    name ``'MegaCam'``.

    Attributes
    ----------
    facility : `CFHT`
        The CFHT facility instance hosting MegaCam.
    filt_names : `list` of `str`
        Names of all MegaCam filters.
    SVO_name : `str`
        SVO instrument name (``'MegaCam'``).
    """

    def __init__(self) -> None:
        Megacam_filt_names = [
            "u",
            "u_1",
            "g",
            "g_1",
            "r",
            "r_1",
            "i",
            "i_1",
            "i_2",
            "z",
            "z_1",
            # "gri",
        ]
        self.SVO_name = "MegaCam"
        super().__init__(CFHT(), Megacam_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords used to calculate the
        zero-point (``'PHOTZP'``)."""
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from the image header.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the image header used to look up the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band, taken directly
            from the ``'PHOTZP'`` header keyword.
        """
        return band_data.load_im()[1]["PHOTZP"]

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for MegaCam.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for MegaCam.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for MegaCam!"
        )


class HSC(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing Subaru/HSC.

    Defines the full set of Hyper Suprime-Cam filters. Uses the SVO
    instrument name ``'HSC'``.

    Attributes
    ----------
    facility : `Subaru`
        The Subaru facility instance hosting HSC.
    filt_names : `list` of `str`
        Names of all HSC filters.
    SVO_name : `str`
        SVO instrument name (``'HSC'``).
    """

    def __init__(self) -> None:
        HSC_filt_names = [
            "g",
            "r",
            "i",
            "z",
            "Y",
            "NB387_filter",
            "NB468_filter",
            "g_filter",
            "NB515_filter",
            "r2_filter",
            "r_filter",
            "NB656_filter",
            "NB718_filter",
            "i_filter",
            "i2_filter",
            "NB816_filter",
            "z_filter",
            "NB921_filter",
            "NB926_filter",
            "IB945_filter",
            "NB973_filter",
            "Y_filter",
        ]
        self.SVO_name = "HSC"
        super().__init__(Subaru(), HSC_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords used to calculate the
        zero-point (``'PHOTZP'``)."""
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from the image header.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the image header used to look up the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band, taken directly
            from the ``'PHOTZP'`` header keyword.
        """
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for HSC.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for HSC.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for HSC!"
        )


class VIS(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing Euclid/VIS.

    Defines the single VIS visible-light filter. Uses the SVO
    instrument name ``'VIS'``.

    Attributes
    ----------
    facility : `Euclid`
        The Euclid facility instance hosting VIS.
    filt_names : `list` of `str`
        Names of all VIS filters.
    SVO_name : `str`
        SVO instrument name (``'VIS'``).
    """

    def __init__(self) -> None:
        VIS_filt_names = ["vis"]
        self.SVO_name = "VIS"
        super().__init__(Euclid(), VIS_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords used to calculate the
        zero-point (``'PHOTZP'``)."""
        return ["PHOTZP"]

    def calc_ZP(self: Self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from the image header.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the image header used to look up the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band, taken directly
            from the ``'PHOTZP'`` header keyword.
        """
        return band_data.load_im()[1]["PHOTZP"]

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for VIS.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for VIS.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for VIS!"
        )


class NISP(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing Euclid/NISP.

    Defines the full set of NISP near-infrared filters. Uses the SVO
    instrument name ``'NISP'``.

    Attributes
    ----------
    facility : `Euclid`
        The Euclid facility instance hosting NISP.
    filt_names : `list` of `str`
        Names of all NISP filters.
    SVO_name : `str`
        SVO instrument name (``'NISP'``).
    """

    def __init__(self) -> None:
        NISP_filt_names = [
            "Y",
            "J",
            "H",
        ]
        self.SVO_name = "NISP"
        super().__init__(Euclid(), NISP_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords used to calculate the
        zero-point (``'PHOTZP'``)."""
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from the image header.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the image header used to look up the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band, taken directly
            from the ``'PHOTZP'`` header keyword.
        """
        return band_data.load_im()[1]["PHOTZP"]

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for NISP.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for NISP.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for NISP!"
        )


class IRAC(Instrument, funcs.Singleton):
    """Singleton `Instrument` representing Spitzer/IRAC.

    Defines the full set of IRAC infrared filters. Uses the SVO
    instrument name ``'IRAC'``.

    Attributes
    ----------
    facility : `Spitzer`
        The Spitzer facility instance hosting IRAC.
    filt_names : `list` of `str`
        Names of all IRAC filters.
    SVO_name : `str`
        SVO instrument name (``'IRAC'``).
    """

    def __init__(self) -> None:
        IRAC_filt_names = [
            "I1",
            "I2",
            "I3",
            "I4",
        ]
        self.SVO_name = "IRAC"
        super().__init__(Spitzer(), IRAC_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        """`list` of `str`: FITS header keywords used to calculate the
        zero-point (``'PHOTZP'``)."""
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        """Calculate the AB magnitude zero-point from the image header.

        Parameters
        ----------
        band_data : `Band_Data_Base`
            Band data providing the image header used to look up the
            zero-point.

        Returns
        -------
        `astropy.units.Quantity`
            The AB magnitude zero-point for this band, taken directly
            from the ``'PHOTZP'`` header keyword.
        """
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        """Construct a model PSF for IRAC.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data for which to construct the model PSF.

        Raises
        ------
        NotImplementedError
            Model PSF construction is not yet implemented for IRAC.
        """
        raise NotImplementedError(
            "Model PSF construction not yet implemented for IRAC!"
        )


# Instrument attributes

# TODO: Generalize this so the user does not
# have to update upon the addition of a new instrument
expected_instr_bands = {
    "ACS_WFC": ACS_WFC().filt_names,
    "WFC3_IR": WFC3_IR().filt_names,
    "NIRCam": NIRCam().filt_names,
    "MIRI": MIRI().filt_names,
    "VISTA": VISTA().filt_names,
    "MegaCam": MegaCam().filt_names,
    "HSC": HSC().filt_names,
    "VIS": VIS().filt_names,
    "NISP": NISP().filt_names,
    "IRAC": IRAC().filt_names,
}

expected_instr_facilities = {
    "ACS_WFC": "HST",
    "WFC3_IR": "HST",
    "NIRCam": "JWST",
    "MIRI": "JWST",
    "VISTA": "Paranal",
    "MegaCam": "CFHT",
    "HSC": "Subaru",
    "VIS": "Euclid",
    "NISP": "Euclid",
    "IRAC": "Spitzer",
}

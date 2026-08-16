# Instrument.py
from __future__ import annotations

from typing import NoReturn, Dict, Any, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Band_Data, Band_Data_Base, Data
    from . import PSF_Base, PSF_Cutout
    from . import Filter, Multiple_Filter

import astropy.units as u
from astropy.io import ascii
import h5py
from astropy.table import Table
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from astroquery.svo_fps import SvoFps
from copy import deepcopy
import json
from abc import ABC, abstractmethod
from typing import Tuple

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import config, galfind_logger
from . import useful_funcs_austind as funcs
from . import PSF_Cutout


class Facility(ABC):
    # Facility class to store the name of the facility
    # and other facility-specific attributes/methods

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
    pass


class JWST(Facility, funcs.Singleton):
    pass

class Paranal(Facility, funcs.Singleton):
    pass

class Spitzer(Facility, funcs.Singleton):
    pass

class Euclid(Facility, funcs.Singleton):
    pass

class CFHT(Facility, funcs.Singleton):
    pass

class Subaru(Facility, funcs.Singleton):
    pass


class Instrument(ABC):

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
            output_str += f"FACILITY:\n"
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
        pass

    @abstractmethod
    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        pass

    def calc_pix_scale(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        pass

    def make_psf(
        self: Self,
        band_data: Band_Data,
        method: str = "default",
        size: u.Quantity = 0.96 * u.arcsec,
    ) -> Type[PSF_Base]:
        method_types = ["default", "empirical", "EPOCHS"]
        assert method in method_types, \
            galfind_logger.critical(
                f"{method=} not in {method_types}!"
            )
        if method == "default":
            return self.make_model_psf(
                band_data,
                size = size,
            )
        elif method == "empirical":
            return PSF_Cutout.from_empirical_psf(band_data)
        else: # method == "EPOCHS":
            if isinstance(self, ACS_WFC):
                return self.make_model_psf(
                    band_data,
                    size = size,
                )
            else:
                epochs_path = f"{config['PSF']['PSF_WORK_DIR']}/EPOCHS_PSFs/{band_data.filt_name}/PSF_Resample_03_{band_data.filt_name}.fits"
                return PSF_Cutout.from_fits(
                    epochs_path,
                    band_data,
                    size = size,
                    origin = "webbpsf",
                )

    @abstractmethod
    def make_model_psf(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 0.96 * u.arcsec,
    ) -> Type[PSF_Base]:
        pass

    # def make_empirical_psf(
    #     self: Self,
    #     band_data: Band_Data,
    # ) -> Type[PSF_Base]:
    #     raise NotImplementedError(
    #         f"Empirical PSF construction not yet implemented for {repr(self)}!"
    #     )

    def get_psf_norm(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 4.0 * u.arcsec,
    ) -> Optional[Tuple[float, float]]:
        return None  

    def get_psf_norm_path(
        self: Self,
        **kwargs: Dict[str, Any],
    ) -> None:
        return None
    
    @staticmethod
    def get_psf_dir(band_data: Band_Data) -> str:
        return f"{config['PSF']['PSF_WORK_DIR']}/{band_data.filt.instrument.__class__.__name__}" + \
            f"/{band_data.version}/{band_data.survey}/{band_data.filt_name}"

    def get_eec_path(
        self: Self,
        band_data: Band_Data,
    ) -> str:
        eec_name = f"EEC_{band_data.filt.filt_name}.h5"
        eec_path = f"{self.get_psf_dir(band_data)}/{eec_name}"
        return eec_path


class NIRCam(Instrument, funcs.Singleton):
    
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
        return []

    def calc_ZP(
        self: Self,
        band_data: Type[Band_Data_Base]
    ) -> u.Quantity:
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
        from . import PSF_Cutout
        # TODO: create PSF_Cutout from WebbPSF instead of loading from file
        #psf_path = f"{psf_dir}/NIRCam/{band_data.filt.filt_name}.fits"
        psf = PSF_Cutout.from_stpsf(
            band_data,
            #psf_path,
            #band_data.filt,
            #size = size,
        )
        return psf

    def get_psf_norm(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 4.0 * u.arcsec,
    ) -> Union[None, Tuple[float, float]]:
        norm_path = self.get_psf_norm_path(band_data = band_data)
        if norm_path is None:
            return None
        else:
            assert Path(norm_path).is_file(), \
                galfind_logger.critical(
                    f"PSF normalization file {norm_path} not found!"
                )
            energy_table = ascii.read(norm_path)
            row = np.argmin(abs(size.to(u.arcsec).value / 2.0 - energy_table["aper_radius"]))
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
        SW_norm_path = f"{config['PSF']['PSF_DIR']}/{self.__class__.__name__}/encircled_energy/Encircled_Energy_SW_ETCv2.txt"
        LW_norm_path = f"{config['PSF']['PSF_DIR']}/{self.__class__.__name__}/encircled_energy/Encircled_Energy_LW_ETCv2.txt"
        filters = {
            "SW": Table.read(SW_norm_path, format = "ascii").colnames[1:],
            "LW": Table.read(LW_norm_path, format = "ascii").colnames[1:],
        }
        if band_data.filt.filt_name in filters["SW"]:
            return SW_norm_path
        elif band_data.filt.filt_name in filters["LW"]:
            return LW_norm_path
        else:
            galfind_logger.warning(f"Filter {band_data.filt.filt_name} not found in any PSF normalization table!")
            return None


class MIRI(Instrument, funcs.Singleton):
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
        return []

    def calc_ZP(self: Self, band_data: Type[Band_Data_Base]) -> u.Quantity:
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
        raise NotImplementedError("Model PSF construction not yet implemented for MIRI!")


class ACS_WFC(Instrument, funcs.Singleton):

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
            #"FR551N",
            "F550M",
            "F606W",
            "FR601N",
            "F625W",
            "FR647M",
            "FR656N",
            "F658N",
            "F660N",
            "FR716N",
            #"POL_UV",
            "G800L",
            #"POL_V",
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
        return ["PHOTFLAM", "PHOTPLAM", "ZEROPNT"] # or 'BUNIT'=MJy/sr
        
    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
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
                f"Using ZEROPNT from header for {band_data.filt_name} " + \
                "instead of PHOTFLAM and PHOTPLAM! Potential ST mags used here!"
            )
        # DO NOT UNCOMMENT! REGULARLY CAUSES ZP ERRORS
        # elif "BUNIT" in im_header:
        #     unit = im_header["BUNIT"].replace(" ", "")
        #     assert unit == "MJy/sr"
        #     ZP = -2.5 * np.log10(
        #         (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
        #     ) + u.Jy.to(u.ABmag)
        else:
            err_message = f"ACS_WFC data for {band_data.filt_name} " + \
                "must contain either 'PHOTFLAM' and 'PHOTPLAM' or 'ZEROPNT'" + \
                "in its header to calculate its ZP!" # or 'BUNIT'=MJy/sr 
            galfind_logger.critical(err_message)
            raise (Exception(err_message))
        return ZP

    def get_psf_norm_path(
        self: Self,
        **kwargs: Dict[str, Any],
    ) -> str:
        return f"{config['PSF']['PSF_DIR']}/{self.__class__.__name__}/encircled_energy/ACS_WFC_EE.txt"

    def get_psf_norm(
        self: Self,
        band_data: Band_Data,
        size: u.Quantity = 4.0 * u.arcsec,
    ) -> Tuple[float, float]:
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
        from . import PSF_Base
        self._make_eec(band_data)
        return PSF_Base(
            self.get_eec_path(band_data),
            name = f"{band_data.filt_name}_model",
            **kwargs,
        )

    def _make_eec(
        self: Self,
        band_data: Band_Data,
    ) -> None:
        eec_path = self.get_eec_path(band_data)
        if not Path(eec_path).is_file():
            galfind_logger.info(
                f"Creating EEC data for {repr(band_data.filt)} at {eec_path}"
            )
            txt_filepath = self.get_psf_norm_path(band_data = band_data)
            assert Path(txt_filepath).is_file(), \
                galfind_logger.critical(
                    f"EE data for {self.__class__.__name__} not found in {txt_filepath}!"
                )
            eec_tab = Table.read(txt_filepath, format = "ascii.commented_header")
            eec_tab = eec_tab[eec_tab["band"] == band_data.filt.filt_name]
            assert len(eec_tab) == 1, \
                galfind_logger.critical(
                    f"EE data for {repr(band_data.filt)} not found in {txt_filepath}!"
                )
            radii = np.zeros(len(eec_tab.colnames) - 1) # * u.arcsec
            eec = np.zeros(len(eec_tab.colnames) - 1)
            for i, col in enumerate(eec_tab.colnames):
                if col != "band":
                    radii[i - 1] = float(col) * 0.05 # * u.arcsec
                    eec[i - 1] = eec_tab[col]
            radii = radii.astype(np.float32)
            eec = eec.astype(np.float32)
            # save eec data to h5 file
            funcs.make_dirs(eec_path)
            with h5py.File(eec_path, "w") as f:
                f.create_dataset("radii", data = radii, compression = "gzip")
                f.create_dataset("eec", data = eec, compression = "gzip")
                galfind_logger.info(
                    f"Saved EEC data for {repr(band_data.filt)} to {eec_path}"
                )
            funcs.change_file_permissions(eec_path)
        else:
            galfind_logger.debug(
                f"EEC data for {repr(band_data.filt)} already exists at {eec_path}!"
            )


class WFC3_IR(Instrument, funcs.Singleton):

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
        return []

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
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
        breakpoint()


class VISTA(Instrument, funcs.Singleton):
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
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        breakpoint()


class MegaCam(Instrument, funcs.Singleton):
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
            #"gri",
        ]
        self.SVO_name = "MegaCam"
        super().__init__(CFHT(), Megacam_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        return band_data.load_im()[1]["PHOTZP"]

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        breakpoint()


class HSC(Instrument, funcs.Singleton):
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
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP
    
    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        breakpoint()


class VIS(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        VIS_filt_names = [
            "vis"
        ]
        self.SVO_name = "VIS"
        super().__init__(Euclid(), VIS_filt_names)

    @property
    def ZP_keys(self) -> List[str]:
        return ["PHOTZP"]

    def calc_ZP(self: Self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        return band_data.load_im()[1]["PHOTZP"]

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        breakpoint()


class NISP(Instrument, funcs.Singleton):
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
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        return band_data.load_im()[1]["PHOTZP"]

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        breakpoint()


class IRAC(Instrument, funcs.Singleton):
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
        return ["PHOTZP"]

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_psf(
        self: Self,
        band_data: Band_Data,
    ) -> Type[PSF_Base]:
        breakpoint()


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


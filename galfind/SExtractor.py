from __future__ import annotations

import numpy as np
import json
import astropy.units as u
import subprocess
from pathlib import Path
import time
import os

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
from typing import Optional, Tuple, Union, NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Band_Data_Base

from . import config, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir

def get_code() -> str:
    # of order 0.1s per call
    return (
        subprocess.check_output("sex --version", shell=True)
        .decode("utf-8")
        .replace("\n", "")
    )

def get_segmentation_path(
    self: Type[Band_Data_Base],
    err_map_type: str,
) -> str:
    seg_dir = f"{config['SExtractor']['SEX_DIR']}/{self.instr_name}/{self.version}/{self.survey}/{err_map_type}/segmentation"
    seg_path = f"{seg_dir}/{self.survey}_{self.filt_name}_{self.filt_name}_sel_cat_{self.version}_seg.fits"
    funcs.make_dirs(seg_path)
    return seg_path

def get_forced_phot_path(
    self: Type[Band_Data_Base],
    err_map_type: str,
    forced_phot_band: Optional[Type[Band_Data_Base]] = None,
) -> str:
    forced_phot_dir = f"{config['SExtractor']['SEX_DIR']}/{self.instr_name}/{self.version}/{self.survey}/{err_map_type}/forced_phot/{funcs.aper_diams_to_str(self.aper_diams)}"
    if forced_phot_band is None:
        select_band_name = self.filt_name
    else:
        select_band_name = forced_phot_band.filt_name
    forced_phot_path = f"{forced_phot_dir}/{self.survey}_{self.filt_name}_{select_band_name}_sel_cat_{self.version}.fits"
    funcs.make_dirs(forced_phot_path)
    return forced_phot_path

def get_err_map(
    self: Type[Band_Data_Base], err_type: str
) -> Union[Tuple[str, int, str], NoReturn]:
    if err_type == "rms_err":
        if self.rms_err_path is not None and self.rms_err_ext is not None:
            pass
        else:
            self._make_rms_err_from_wht()
        return self.rms_err_path, self.rms_err_ext, "MAP_RMS"
            # raise (
            #     Exception(
            #         f"No rms_err map available for {self.filt.band_name}"
            #     )
            # )
    elif err_type == "wht":
        if self.wht_path is not None and self.wht_ext is not None:
            pass
        else:
            self._make_wht_from_rms_err()
            # raise (
            #     Exception(f"No wht map available for {self.filt.band_name}")
            # )
        return self.wht_path, self.wht_ext, "MAP_WEIGHT"
    else:
        raise (
            Exception(f"err_type must be 'rms_err' or 'wht', not {err_type}")
        )

@run_in_dir(path=config["DEFAULT"]["GALFIND_DIR"])
def segment(
    self: Type[Band_Data_Base],
    err_type: str = "rms_err",
    config_name: str = "default.sex",
    params_name: str = "default.param",
    overwrite: bool = False,
) -> str:
    
    sex_config_path = f"{config['SExtractor']['SEX_CONFIG_DIR']}/{config_name}"
    params_path = f"{config['SExtractor']['SEX_CONFIG_DIR']}/{params_name}"

    err_map_path, err_map_ext, err_map_type = get_err_map(self, err_type)
    seg_path = get_segmentation_path(self, err_map_type)
    
    if not Path(seg_path).is_file() or overwrite:
        galfind_logger.info(
            "Making SExtractor seg/bkg maps for "
            f"{self.survey} {self.version} {self.filt_name}"
        )
        # update the SExtractor params file at runtime
        # to include the correct number of aperture diameters
        update_sex_params_aper_diam_len(len(self.aper_diams), params_path)
        pix_aper_diams = (
            str(
                [
                    np.round(pix_aper_diam, 2)
                    for pix_aper_diam in (self.aper_diams / self.pix_scale).to(u.dimensionless_unscaled).value
                ]
            )
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )
        input = [
            "./make_seg_map.sh",
            config["SExtractor"]["SEX_DIR"],
            self.im_path,
            str(self.pix_scale.value),
            str(self.ZP),
            self.instr_name,
            self.survey,
            self.filt_name,
            self.version,
            err_map_path,
            str(err_map_ext),
            err_map_type,
            str(self.im_ext),
            sex_config_path,
            params_path,
            pix_aper_diams,
        ]
        # SExtractor bash script python wrapper
        galfind_logger.debug(input)
        process = subprocess.Popen(input)
        process.wait()
        funcs.change_file_permissions(seg_path)
    return seg_path


def update_sex_params_aper_diam_len(aper_diam_length: int, params_path: str) -> None:
    with open(params_path, "r") as f:
        lines = f.readlines()
        f.close()
    new_lines = [
        line
        for i, line in enumerate(lines)
        if (
            "MAG_APER" not in line
            and "MAGERR_APER" not in line
            and "FLUX_APER" not in line
            and "FLUXERR_APER" not in line
        )
        or "(1)" in line
    ]
    for name in ["MAG_APER", "MAGERR_APER", "FLUX_APER", "FLUXERR_APER"]:
        aper_loc = [i for i, line in enumerate(new_lines) if name in line]
        [
            new_lines.insert(aper_loc[0] + 1, f"{name}({str(n)})\n")
            for n in reversed(range(2, aper_diam_length + 1))
        ]
    with open(params_path, "w") as f:
        f.writelines(new_lines)
        f.close()


@run_in_dir(path=config["DEFAULT"]["GALFIND_DIR"])
def perform_forced_phot(
    self: Type[Band_Data_Base],
    forced_phot_band: Type[Band_Data_Base],
    err_type: str = "rms_err",
    config_name: str = "default.sex",
    params_name: str = "default.param",
    timed: bool = True,
    overwrite: bool = False,
) -> str:

    sex_config_path = f"{config['SExtractor']['SEX_CONFIG_DIR']}/{config_name}"
    params_path = f"{config['SExtractor']['SEX_CONFIG_DIR']}/{params_name}"

    # make forced photometry catalogue
    start = time.time()
    err_map_path, err_map_ext, err_map_type = get_err_map(self, err_type)
    select_band_err_map_path, select_band_map_ext, select_band_map_type = (
        get_err_map(forced_phot_band, err_type)
    )
    assert err_map_type == select_band_map_type, galfind_logger.critical(
        f"{err_map_type=}!={select_band_map_type=}"
    )
    forced_phot_path = get_forced_phot_path(
        self, err_map_type, forced_phot_band
    )
    if not Path(forced_phot_path).is_file() or overwrite:
        # check whether the image of the forced photometry band and sextraction band have the same shape
        assert (
            self.data_shape == forced_phot_band.data_shape
        ), galfind_logger.critical(
            f"{self.data_shape=}!={forced_phot_band.data_shape=}"
        )
        # update the SExtractor params file at runtime
        # to include the correct number of aperture diameters
        update_sex_params_aper_diam_len(len(self.aper_diams), params_path)
        pix_aper_diams = (
            str(
                [
                    np.round(pix_aper_diam, 2)
                    for pix_aper_diam in (self.aper_diams / self.pix_scale).to(u.dimensionless_unscaled).value
                ]
            )
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )
        input = [
            "./make_sex_cat.sh",
            config["SExtractor"]["SEX_DIR"],
            self.im_path,
            str(self.pix_scale.value),
            str(self.ZP),
            self.instr_name,
            self.survey,
            self.filt_name,
            self.version,
            forced_phot_band.filt_name,
            forced_phot_band.im_path,
            err_map_path,
            str(err_map_ext),
            str(self.im_ext),
            select_band_err_map_path,
            str(forced_phot_band.im_ext),
            err_map_type,
            str(select_band_map_ext),
            sex_config_path,
            params_path,
            pix_aper_diams,
        ]
        # SExtractor bash script python wrapper
        galfind_logger.debug(input)
        process = subprocess.Popen(input)
        process.wait()
        os.rename(forced_phot_path.replace(f"/{funcs.aper_diams_to_str(self.aper_diams)}", ""), forced_phot_path)
        finish_message_prefix = "Made"
    else:
        finish_message_prefix = "Loaded"

    finish_message = (
        f"{finish_message_prefix} SExtractor catalogue for "
        + f"{self.survey} {self.version} {repr(self)}"
        + f" using {repr(forced_phot_band)}!"
    )
    if timed:
        end = time.time()
        finish_message += f" ({end - start:.1f}s)"
    galfind_logger.debug(finish_message)

    forced_phot_args = \
        {
            "forced_phot_band": forced_phot_band,
            "err_type": err_type,
            "method": get_code(),
            "config_name": config_name,
            "params_name": params_name,
            "id_label": "NUMBER",
            "ra_label": "ALPHA_J2000",
            "dec_label": "DELTA_J2000",
            "ra_unit": u.deg,
            "dec_unit": u.deg,
        }
    return forced_phot_path, forced_phot_args

    # if self.forced_phot_band not in self.instrument.band_names:
    #     sextractor_bands = [band for band in self.instrument] + [
    #         self.forced_phot_band
    #     ]
    # else:
    #     sextractor_bands = [band for band in self.instrument]
    # sex_cat_type = (
    #     subprocess.check_output("sex --version", shell=True)
    #     .decode("utf-8")
    #     .replace("\n", "")
    # )
    # # of order 0.1s per call
    # galfind_logger.debug(
    #     "'subprocess.check_output()' takes of order 0.1s per call!"
    # )
    # else:
    #     sextract = False
    #     self.sex_cat_types[band_name] = (
    #         f"{forced_phot_code} v{globals()[forced_phot_code].__version__}"
    #     )

from __future__ import annotations

import numpy as np
import json
import subprocess

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Band_Data_Base

from . import config, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir

def get_sextractor_seg_path(self, err_type: str = "rms_err") -> str:
    pass

@run_in_dir(path=config["DEFAULT"]["GALFIND_DIR"])
def segment_sextractor(
    self: Type[Band_Data_Base],
    err_type: str = "rms_err",
    sex_config_path: str = config["SExtractor"]["CONFIG_PATH"],
    params_path: str = config["SExtractor"]["PARAMS_PATH"],
) -> str:
    # load relevant err map paths, preferring rms_err maps if available
    if err_type == "rms_err":
        if self.rms_err_path is not None and self.rms_err_ext is not None:
            err_map_path = self.rms_err_path
            err_map_ext = self.rms_err_ext
            err_map_type = "MAP_RMS"
        else:
            raise(Exception(f"No rms_err map available for {self.filt.band_name}"))
    elif err_type == "wht":
        if self.wht_path is not None and self.wht_ext is not None:
            err_map_path = self.wht_path
            err_map_ext = self.wht_ext
            err_map_type = "MAP_WEIGHT"
        else:
            raise(Exception(f"No wht map available for {self.filt.band_name}"))
    else:
        raise(Exception(f"err_type must be 'rms_err' or 'wht', not {err_type}"))

    # insert specified aperture diameters from config file
    as_aper_diams = json.loads(config.get("SExtractor", "APERTURE_DIAMS"))
    # update the SExtractor params file to include the correct number of aperture diameters
    update_sex_params_aper_diam_len(len(as_aper_diams))
    pix_aper_diams = (
        str(
            [
                np.round(pix_aper_diam, 2)
                for pix_aper_diam in as_aper_diams
                / self.pix_scale.value
            ]
        )
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
    )

    seg_dir = f"{config['SExtractor']['SEX_DIR']}/{self.instr_name}/{self.version}/{self.survey}/{err_map_type}"
    seg_path = f"{seg_dir}/{self.survey}_{self.filt_name}_{self.filt_name}_sel_cat_{self.version}_seg.fits"
    funcs.make_dirs(seg_path)

    print(
        [
            "./make_seg_map.sh",
            config["SExtractor"]["SEX_DIR"],
            self.im_path,
            str(self.pix_scale.value),
            str(self.ZP),
            self.instr_name,
            self.survey,
            self.filt.band_name,
            self.version,
            err_map_path,
            str(err_map_ext),
            err_map_type,
            str(self.im_ext),
            sex_config_path,
            params_path,
            pix_aper_diams,
        ]
    )

    # SExtractor bash script python wrapper
    process = subprocess.Popen(
        [
            "./make_seg_map.sh",
            config["SExtractor"]["SEX_DIR"],
            self.im_path,
            str(self.pix_scale.value),
            str(self.ZP),
            self.instr_name,
            self.survey,
            self.filt.band_name,
            self.version,
            err_map_path,
            str(err_map_ext),
            err_map_type,
            str(self.im_ext),
            sex_config_path,
            params_path,
            pix_aper_diams,
        ]
    )
    process.wait()
    galfind_logger.info(
        f"Made seg/bkg maps for {self.survey} {self.version}" + \
        f" {self.filt.band_name} using config = {sex_config_path} and {err_map_type}"
    )
    funcs.change_file_permissions(seg_path)
    return seg_path

def update_sex_params_aper_diam_len(aper_diam_length: int):
    with open(config["SExtractor"]["PARAMS_PATH"], "r") as f:
        lines = f.readlines()
        f.close()
    new_lines = [line for i, line in enumerate(lines) if ("MAG_APER" not in line and "MAGERR_APER" not in line and "FLUX_APER" not in line and "FLUXERR_APER" not in line) or "(1)" in line]
    for name in ["MAG_APER", "MAGERR_APER", "FLUX_APER", "FLUXERR_APER"]:
        aper_loc = [i for i, line in enumerate(new_lines) if name in line]
        [new_lines.insert(aper_loc[0] + 1, f"{name}({str(n)})\n") for n in reversed(range(2, aper_diam_length + 1))]
    with open(config["SExtractor"]["PARAMS_PATH"], "w") as f:
        f.writelines(new_lines)
        f.close()

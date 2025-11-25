
import os
import numpy as np
import galfind
from galfind import config
from galfind import useful_funcs_austind as funcs
from pathlib import Path

def get_depth_dir(galfind_work_dir, survey, version, instrument_names):
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/Depths/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_eazy_dir(galfind_work_dir, survey, version, instrument_names):
    instrument_name = "+".join(instrument_names)
    out_dirs = []
    for subdir in ["input", "output"]:
        out_dirs.append(f"{galfind_work_dir}/EAZY/{subdir}/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_mask_dir(galfind_work_dir, survey):
    return np.array([f"{galfind_work_dir}/Masks/{survey}"])

def get_sex_dir(galfind_work_dir, survey, version, instrument_names):
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/SExtractor/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_stacked_images_dir(galfind_work_dir, survey, version, instrument_names):
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/Stacked_Images/{version}/{instrument_name}/{survey}")
    return np.array(out_dirs)

def find_target_dir(galfind_work_dir, survey, version, instrument_names, keyword):
    if keyword == "Depths":
        return get_depth_dir(galfind_work_dir, survey, version, instrument_names)
    elif keyword == "EAZY":
        return get_eazy_dir(galfind_work_dir, survey, version, instrument_names)
    elif keyword == "Masks":
        return get_mask_dir(galfind_work_dir, survey)
    elif keyword == "SExtractor":
        return get_sex_dir(galfind_work_dir, survey, version, instrument_names)
    elif keyword == "Stacked_Images":
        return get_stacked_images_dir(galfind_work_dir, survey, version, instrument_names)
    else:
        raise ValueError(f"Keyword {keyword} not recognised")

def main(target_galfind_work, symlink_galfind_work, survey, version, instrument_names, keywords):
    for keyword in keywords:
        target_dirs = find_target_dir(target_galfind_work, survey, version, instrument_names, keyword)
        for target_dir in target_dirs:
            target_paths = [str(path) for path in Path(target_dir).rglob("*") if path.is_file()]
            symlink_paths = [path.replace(target_galfind_work, symlink_galfind_work) for path in target_paths]
            for target_path, symlink_path in zip(target_paths, symlink_paths):
                funcs.symlink(target_path, symlink_path)

if __name__ == "__main__":

    survey = "JADES-DR3-GS-North"
    version = "v13"
    instrument_names = ["ACS_WFC", "NIRCam"]
    target_dir = galfind.config["DEFAULT"]["GALFIND_WORK"]

    symlink_dir = "/raid/scratch/work/hthomas/GALFIND_WORK"
    dirs_to_link = ["Depths", "EAZY", "Masks", "SExtractor", "Stacked_Images"]

    main(target_dir, symlink_dir, survey, version, instrument_names, dirs_to_link)
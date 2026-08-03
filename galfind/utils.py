
import os
import numpy as np
from astropy import units as u
import shutil
from pathlib import Path

from . import config
from . import useful_funcs_austind as funcs
#from .Data import morgan_version_to_dir_dict

def get_data_dir(galfind_data_dir, survey, version, instrument_names, pix_scale = 0.03 * u.arcsec, version_to_dir_dict = None):
    from . import Instrument
    out_dirs = []
    for instrument_name in instrument_names:
        # determine facility name from instrument name
        instr = [cls for cls in funcs.all_subclasses(Instrument) if cls.__name__ == instrument_name]
        assert len(instr) == 1, f"Instrument {instrument_name} not recognised"
        instr = instr[0]()
        facility_name = instr.facility.SVO_name.lower()
        if version_to_dir_dict is None:
            version_ = version
        else:
            assert version in version_to_dir_dict, f"Version {version} not recognised"
            version_ = version_to_dir_dict[version]
        out_dirs.append(
            f"{galfind_data_dir}/{facility_name}/{survey}/{instrument_name}/" + \
            f"{version_}/{pix_scale.to(u.marcsec).value:.0f}mas"
        )
    return np.array(out_dirs)

def get_cat_dir(galfind_work_dir, survey, version, instrument_names):
    instrument_name = "+".join(instrument_names)
    return np.array([f"{galfind_work_dir}/Catalogues/{version}/{instrument_name}/{survey}"])

def get_depth_dir(galfind_work_dir, survey, version, instrument_names):
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/Depths/{instrument_name}/{version}/{survey}")
    #breakpoint()
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

def find_target_dir(galfind_dir, survey, version, instrument_names, keyword):
    if keyword == "Data":
        return get_data_dir(galfind_dir, survey, version, instrument_names)
    if keyword == "Catalogues":
        return get_cat_dir(galfind_dir, survey, version, instrument_names)
    if keyword == "Depths":
        return get_depth_dir(galfind_dir, survey, version, instrument_names)
    elif keyword == "EAZY":
        return get_eazy_dir(galfind_dir, survey, version, instrument_names)
    elif keyword == "Masks":
        return get_mask_dir(galfind_dir, survey)
    elif keyword == "SExtractor":
        return get_sex_dir(galfind_dir, survey, version, instrument_names)
    elif keyword == "Stacked_Images":
        return get_stacked_images_dir(galfind_dir, survey, version, instrument_names)
    else:
        raise ValueError(f"Keyword {keyword} not recognised")

def symlink(target_galfind_dir, symlink_galfind_dir, survey, version, instrument_names, keywords):
    for keyword in keywords:
        target_dirs = find_target_dir(target_galfind_dir, survey, version, instrument_names, keyword)
        for target_dir in target_dirs:
            target_paths = [str(path) for path in Path(target_dir).rglob("*") if path.is_file()]
            symlink_paths = [path.replace(target_galfind_dir, symlink_galfind_dir) for path in target_paths]
            for target_path, symlink_path in zip(target_paths, symlink_paths):
                funcs.symlink(target_path, symlink_path)

def unlink(target_galfind_dir, survey, version, instrument_names, keywords):
    for keyword in keywords:
        target_dirs = find_target_dir(target_galfind_dir, survey, version, instrument_names, keyword)
        for target_dir in target_dirs:
            target_paths = [str(path) for path in Path(target_dir).rglob("*") if path.is_file()]
            for symlink_path in target_paths:
                if os.path.islink(symlink_path):
                    os.unlink(symlink_path)

 
def _is_up_to_date(target_path, dest_size, dest_mtime):
    """Compare a source file against a known destination size/mtime."""
    src_stat = os.stat(target_path)
    return src_stat.st_size == dest_size and int(src_stat.st_mtime) <= int(dest_mtime)
 
def _remote_copy(ssh_client, sftp_client, target_path, dest_path, force=False):
    """Create parent dirs on the remote host, then copy the file over via SFTP
    unless an up-to-date copy already exists there."""
    parent = str(Path(dest_path).parent)
    cmd = f"mkdir -p {parent!r}"
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        err = stderr.read().decode().strip()
        raise RuntimeError(f"Failed to create remote dir {parent!r}: {err}")
 
    if not force:
        try:
            dest_stat = sftp_client.stat(dest_path)
            if _is_up_to_date(target_path, dest_stat.st_size, dest_stat.st_mtime):
                return False  # skipped, already up to date
        except FileNotFoundError:
            pass
    try:
        sftp_client.put(target_path, dest_path)
    except Exception as e:
        print(f"Failed to copy {target_path!r} to {dest_path!r}: {e}")
        return None
    src_stat = os.stat(target_path)
    sftp_client.utime(dest_path, (src_stat.st_atime, src_stat.st_mtime))
    return True  # copied
 
def _local_copy(target_path, dest_path, force=False):
    """Copy a file locally, recreating any needed subdirectories, unless an
    up-to-date copy already exists at the destination."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
 
    if not force and dest_path.exists():
        dest_stat = dest_path.stat()
        if _is_up_to_date(target_path, dest_stat.st_size, dest_stat.st_mtime):
            return False  # skipped, already up to date
 
    shutil.copy2(target_path, dest_path)
    return True  # copied


if __name__ == "__main__":

    # #survey = "COSMOS-Web-1A"
    # version = "v11" #"mosaic_1084_wispnathan_v2"
    # instrument_names = ["ACS_WFC", "NIRCam"]
    # target_dir = "/raid/scratch/work/jarcidia/GALFIND_WORK" #galfind.config["DEFAULT"]["GALFIND_WORK"]

    # symlink_dir = "/raid/scratch/work/austind/GALFIND_WORK"
    # dirs_to_link = ["Masks"] #["Depths", "EAZY", "Masks", "SExtractor", "Stacked_Images"]
    # type = "symlink"
    # for survey in [f"COSMOS-Web-{x}{letter}" for x in range(0,8) for letter in ["A", "B"]]:
    #     if type == "symlink":
    #         symlink(target_dir, symlink_dir, survey, version, instrument_names, dirs_to_link)
    #     elif type == "unlink":
    #         unlink(symlink_dir, survey, version, instrument_names, dirs_to_link)
    #     else:
    #         raise ValueError(f"Type {type} not recognised")
    pass


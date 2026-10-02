"""JWST NIRCam raw data reduction and calibration.

Wraps MAST queries and JWST calibration pipeline Stages 1-3 (detector-level,
image-level, and association-based resampling) for NIRCam data.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import vstack
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.patches import Polygon
from numpy.typing import NDArray
from tqdm import tqdm

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from jwst.pipeline import JWSTPipeline

    from . import Instrument

from .. import config, galfind_logger
from ..utils import useful_funcs_austind as funcs
from ..utils.decorators import log_time, run_in_self_dir
from ..utils.exceptions import (
    ExternalToolError,
    GalfindError,
    InvalidOptionError,
    MissingDataError,
    MissingFileError,
)
from .Data import Data
from .Instrument import NIRCam

CURL_OUTPUT_RE = re.compile(r"--output \./\$\{DOWNLOAD_FOLDER\}'([^']+)'")
FOLDER_RE = re.compile(r"^FOLDER=(.+)$", re.MULTILINE)
NIRCAM_DETECTOR_RE = re.compile(
    r"_(nrca[1-4]|nrcalong|nrcb[1-4]|nrcblong)_uncal\.fits$"
)

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs"
    / "reduction_versions.json"
)


def reduction_version_key(
    stage1_steps: Dict[str, Any],
    stage2_steps: Dict[str, Any],
    stage3_steps: Dict[str, Any],
    jwst_version: str,
    pmap: str,
) -> str:
    """Deterministic key for a reduction's exact configuration.

    Two calls with structurally-equal `steps` dicts (regardless of key
    order), `jwst_version`, and `pmap` always produce the same key, so
    `get_or_register_reduction_version` can tell whether this exact
    configuration has already been aliased.
    """
    payload = {
        "stage1_steps": stage1_steps,
        "stage2_steps": stage2_steps,
        "stage3_steps": stage3_steps,
        "jwst_version": jwst_version,
        "pmap": pmap,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def get_or_register_reduction_version(
    key: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> str:
    """The ``"vN"`` alias for `key`, registering a new one if not yet seen.

    `registry_path` is a JSON file mapping ``{key: alias}``, committed
    to the GALFIND repo so every user resolves the same configuration
    to the same alias regardless of who ran the reduction. A
    configuration not yet in the registry is auto-registered under the
    next unused ``"vN"`` (``max`` existing N + 1) and the registry file
    is updated on disk immediately, so a concurrent reduction with a
    different new configuration doesn't collide with it.
    """
    registry_path = Path(registry_path)
    if registry_path.is_file():
        registry: Dict[str, str] = json.loads(registry_path.read_text())
    else:
        registry = {}

    if key in registry:
        return registry[key]

    existing_numbers = [
        int(alias[1:])
        for alias in registry.values()
        if alias.startswith("v") and alias[1:].isdigit()
    ]
    alias = f"v{max(existing_numbers, default=0) + 1}"

    registry[key] = alias
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )
    return alias


def expected_uncal_sizes(downloads_dir: Path) -> Dict[str, int]:
    """Expected file sizes (bytes) for this PID's UNCAL products.

    Read from the MAST product manifest(s) (``*_uncals.fits``) saved
    by `Raw_JWST_Data.query_mast` in `downloads_dir`. Returns an empty
    `dict` if no manifest is found, in which case completeness can't
    be checked and file existence alone is used.
    """
    from astropy.table import Table

    sizes: Dict[str, int] = {}
    for manifest_path in glob.glob(str(downloads_dir / "*_uncals.fits")):
        table = Table.read(manifest_path)
        if "productFilename" in table.colnames and "size" in table.colnames:
            sizes.update(
                {
                    os.path.basename(row["productFilename"]): int(row["size"])
                    for row in table
                }
            )
    return sizes


def existing_uncal_filenames(
    downloads_dir: Path,
    later_stage_base_dir: Optional[Path] = None,
    input_crds: int = 1584,
    cache: Optional[Dict[Tuple[str, str, int], bool]] = None,
) -> Set[str]:
    """Filenames of UNCAL files that don't need to be (re)downloaded
    for this PID.

    Checks both files still nested under ``downloads/`` and files
    already moved to a sibling ``uncals/`` directory. A file whose
    size doesn't match MAST's expected size (from the saved product
    manifest, if any) is treated as not yet downloaded, so a
    truncated or partial download is queued for re-download rather
    than silently left corrupt.

    If `later_stage_base_dir` is given, a filename that isn't present
    locally is nonetheless treated as not needing (re)download when a
    later pipeline stage (RATE, CAL, or stage 3 science) has already
    produced output for it under that directory - e.g. because its
    UNCAL file was intentionally removed by
    `Raw_JWST_Data.remove_uncals_with_rate` once no longer needed.
    Pass `None` (default) to check UNCAL completeness only. `cache` is
    forwarded to `later_stage_output_exists` - see its docstring.
    """
    expected_sizes = expected_uncal_sizes(downloads_dir)
    local_files = glob.glob(
        str(downloads_dir / "*/*/*/*_uncal.fits")
    ) + glob.glob(str(downloads_dir / ".." / "uncals" / "*_uncal.fits"))

    complete = set()
    for f in local_files:
        filename = os.path.basename(f)
        expected = expected_sizes.get(filename)
        if expected is None or os.path.getsize(f) == expected:
            complete.add(filename)

    if later_stage_base_dir is not None:
        # each check opens up to 3 FITS files (RATE/CAL/science) to
        # verify pipeline completeness, which for a programme with
        # thousands of exposures and no local UNCAL files left (e.g.
        # after `remove_uncals_with_rate`) takes long enough to look
        # hung without a progress indicator
        missing = [f for f in expected_sizes if f not in complete]
        for filename in tqdm(
            missing,
            desc="Checking downstream RATE/CAL/science output for "
            "already-processed UNCAL files",
            disable=galfind_logger.getEffectiveLevel() > logging.INFO,
        ):
            if later_stage_output_exists(
                filename,
                later_stage_base_dir,
                input_crds=input_crds,
                cache=cache,
            ):
                complete.add(filename)
    return complete


@contextlib.contextmanager
def silence_stdio():
    """Context manager that redirects the process's stdout/stderr file
    descriptors to `os.devnull` for the duration of the block,
    restoring them afterward.

    Redirects at the file-descriptor level (not just `sys.stdout`), so
    it also catches output from C extensions and logging handlers
    already bound to the original stream objects - both common
    sources of the JWST pipeline's console chatter. Intended for use
    inside a worker process (e.g. a `multiprocessing.Pool` task), so
    it doesn't affect a caller's own output (e.g. a `tqdm` bar) in the
    parent process.
    """
    # flush anything already buffered by Python's own stdout/stderr
    # objects before swapping the underlying fd, and again before
    # restoring it - otherwise buffered output written during the
    # block flushes only once the original fd is back, leaking
    # through despite the redirect
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


def stage_output_path(
    filename: str, output_dir: str, output_suffix: str
) -> Path:
    """The output path a pipeline stage would produce for `filename`.

    Shared by `Raw_JWST_Data.run` (to pre-filter already-complete files
    before dispatching work to a `multiprocessing.Pool`) and
    `Raw_JWST_Data._call_stage` (which re-checks it as a safety net).
    """
    input_suffix = filename.split("_")[-1]
    out_filename = filename.split("/")[-1].replace(input_suffix, output_suffix)
    return Path(f"{output_dir}/{out_filename}.fits")


def asn_product_name(asn_file: str) -> str:
    """The Level 3 product name declared in `asn_file`.

    Image3Pipeline names its output files from this (e.g. an i2d file
    is written to ``f"{name}_i2d.fits"``), rather than from the
    association filename itself - unlike stage 1/2, whose per-exposure
    output filenames are a suffix-replace on their input filename (see
    `stage_output_path`), so a stage 3 output path can't be derived
    without reading the association.
    """
    with open(asn_file) as f:
        return json.load(f)["products"][0]["name"]


def stage3_output_paths(
    asn_files: List[str],
    output_dir: str,
    cache: Optional[Dict[str, Dict[str, Path]]] = None,
) -> Dict[str, Path]:
    """Expected Image3Pipeline i2d.fits output path for each of `asn_files`.

    Keyed by association filename. `cache`, if given, is a `dict` of
    ``{output_dir: {asn_file: output_path}}`` (e.g.
    `Raw_JWST_Data.stage3_output_paths`), keyed first by `output_dir`
    (the ``science_{input_crds}`` directory outputs for this CRDS
    context are written to) so the same association isn't re-read by a
    later `Raw_JWST_Data.run_stage3` call against the same output
    directory - e.g. one made per ``asn_file_search_dir`` in a loop.
    """
    dir_cache = cache.setdefault(output_dir, {}) if cache is not None else {}
    result = {}
    for asn_file in asn_files:
        if asn_file not in dir_cache:
            dir_cache[asn_file] = Path(output_dir) / (
                f"{asn_product_name(asn_file)}_i2d.fits"
            )
        result[asn_file] = dir_cache[asn_file]
    return result


def parse_i2d_filename(i2d_file: Union[str, Path]) -> Tuple[str, str]:
    """(survey, filter_name) parsed from an Image3Pipeline i2d filename.

    Image3Pipeline names its output ``f"{product_name}_i2d.fits"``, and
    `run_stage3`'s associations declare `product_name` as
    ``f"{pointing}-{filter}"`` (see `asn_product_name`) - so the filter
    is the segment after the *last* hyphen, and the survey (pointing)
    name is everything before it. Standard JWST filter names never
    contain a hyphen, so this split is unambiguous.
    """
    stem = Path(i2d_file).name
    if stem.endswith("_i2d.fits"):
        stem = stem[: -len("_i2d.fits")]
    survey, sep, filt_name = stem.rpartition("-")
    if not sep:
        raise ValueError(
            f"Can't parse survey/filter from i2d_file={str(i2d_file)!r}: "
            "expected '<survey>-<filter>_i2d.fits'."
        )
    return survey, filt_name


def i2d_pixel_scale(
    i2d_file: Union[str, Path], im_ext_name: str = "SCI"
) -> u.Quantity:
    """True on-sky pixel scale of an i2d mosaic, read from its WCS.

    Image3Pipeline's output pixel scale is set by resampling parameters
    (e.g. ``resample.pixel_scale_ratio``), not a fixed value, so it has
    to be read from the file rather than assumed - `Data.pipeline`
    defaults to 30mas for NIRCam, but a given reduction may not be.
    """
    with fits.open(i2d_file) as hdul:
        wcs = WCS(hdul[im_ext_name].header)
    scales = proj_plane_pixel_scales(wcs) * wcs.wcs.cunit[0]
    return np.mean(scales).to(u.arcsec)


def wisp_subtracted_filename(filename: str) -> str:
    """`filename`'s wisp-subtracted counterpart, if one exists.

    `subtract_wisps.subtract_Sunnquist24_wisps` writes its output
    alongside the input 'cal' file with a `'_wisp'` suffix inserted
    before the extension (e.g. ``..._cal.fits`` ->
    ``..._cal_wisp.fits``), and only for the wisp-affected detectors
    (nrca3/nrca4/nrcb3/nrcb4). Returns `filename` unchanged if no such
    file exists, e.g. for other detectors or if wisp subtraction
    hasn't been run.
    """
    wisp_filename = filename.replace(".fits", "_wisp.fits")
    return wisp_filename if os.path.isfile(wisp_filename) else filename


def is_complete_pipeline_output(path: Union[str, Path]) -> bool:
    """Whether `path` is a complete, readable JWST pipeline output.

    `stpipe`'s ``save_results=True`` writes a datamodel's metadata as
    an ``ASDF`` extension *last*, after the SCI/ERR/DQ (and, for
    Detector1, VAR_POISSON/VAR_RNOISE) image extensions - so a process
    killed mid-write (walltime limit, OOM-kill, node failure, ...)
    leaves a file that exists and opens fine, but is missing that
    extension. `Path.is_file()` alone can't tell that file apart from
    a genuinely finished one, which is exactly the failure mode from
    `self.pid=2514`: two RATE files truncated mid-write by an earlier,
    interrupted `run_stage1` were treated as "already done" by every
    existence-only check below, silently skipped on every subsequent
    run, and only surfaced once `run_stage2` tried to read their
    (missing) metadata.

    Used everywhere this module decides a pipeline stage's output (or,
    transitively, an UNCAL file's downstream RATE/CAL/science product)
    already exists, so a truncated file is instead treated as not yet
    produced and gets automatically reprocessed - no manual
    intervention needed to pick it back up.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with fits.open(path) as hdul:
            return "ASDF" in hdul
    except Exception:
        # any read failure (truncated file, corrupt header, ...) means
        # this isn't a usable, complete output
        return False


STAGE_SUFFIX_ORDER = ("rate", "cal", "science")


def later_stage_output_exists(
    filename: str,
    base_dir: Path,
    input_crds: int = 1584,
    cache: Optional[Dict[Tuple[str, str, int], bool]] = None,
) -> bool:
    """Whether a later pipeline stage has already produced output for
    the UNCAL file `filename`.

    Checks the RATE, CAL, and stage 3 ("science") output directories
    under `base_dir` (a programme's `Raw_JWST_Data.folder_name`), using
    the same input-filename -> output-filename convention as
    `stage_output_path`. Used to avoid re-downloading or re-moving an
    UNCAL file that was intentionally removed once no longer needed
    (see `Raw_JWST_Data.remove_uncals_with_rate`).

    Checked with a bare existence check, not `is_complete_pipeline_output`:
    the latter opens every candidate file to verify its ASDF metadata,
    which for a programme with thousands of exposures dominates this
    check's runtime. The tradeoff is that a RATE/CAL/science file left
    truncated by an interrupted run is (wrongly) treated as complete
    here, so if its UNCAL input is also missing locally, that UNCAL
    won't be re-downloaded to redo it - unlike `remove_if_next_stage_exists`
    (used by `remove_uncals_with_rate` to decide what's safe to *delete*),
    which still uses the full `is_complete_pipeline_output` check, since
    getting that one wrong loses the UNCAL input permanently.

    `cache`, if given, is a `dict` shared across calls (e.g.
    `Raw_JWST_Data._later_stage_cache`) that this result is read from
    and written to, so the same file's RATE/CAL/science output isn't
    re-checked by every caller that needs to know - e.g.
    `Raw_JWST_Data.query_mast` and `Raw_JWST_Data.move_uncals`, called
    back-to-back for the same programme with nothing in between that
    could change the answer.
    """
    key = (filename, str(base_dir), input_crds)
    if cache is not None and key in cache:
        return cache[key]
    result = any(
        stage_output_path(
            filename,
            str(Path(base_dir) / f"{suffix}_{input_crds}"),
            suffix,
        ).is_file()
        for suffix in STAGE_SUFFIX_ORDER
    )
    if cache is not None:
        cache[key] = result
    return result


def split_curl_script(
    text: str,
) -> Tuple[str, List[Tuple[Optional[str], str]], str]:
    """Split a MAST bundle curl script into header, per-file blocks, footer.

    Each block is the self-contained chunk of the script (the
    ``cat <<EOT ... EOT`` announcement plus the following ``curl``
    command) responsible for downloading a single file, paired with
    that file's basename.
    """
    lines = text.splitlines(keepends=True)
    body_start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("cat <<EOT") or line.startswith("curl ")
        ),
        len(lines),
    )
    header = "".join(lines[:body_start])

    blocks: List[Tuple[Optional[str], str]] = []
    current: List[str] = []
    last_curl_idx = body_start - 1
    for i, line in enumerate(lines[body_start:], start=body_start):
        current.append(line)
        if line.startswith("curl "):
            match = CURL_OUTPUT_RE.search(line)
            filename = os.path.basename(match.group(1)) if match else None
            blocks.append((filename, "".join(current)))
            current = []
            last_curl_idx = i
    footer = "".join(lines[last_curl_idx + 1 :])
    return header, blocks, footer


def extract_curl_commands(
    script_paths: List[Path],
) -> List[Tuple[str, str, str]]:
    """Extract standalone, one-file-per-command curl commands.

    MAST bundle scripts reference a shared ``${DOWNLOAD_FOLDER}`` bash
    variable set once near the top of the script; this resolves that
    variable per script (from its ``FOLDER=`` line) so each file's
    curl command can be run as its own subprocess, one file at a time,
    rather than executing the whole script as a single opaque process.

    Files referenced by more than one input script are only included
    once, so the same file is never downloaded concurrently by
    multiple workers.

    Returns
    -------
    `list` of `tuple`
        ``(filename, curl_command, download_folder)`` triples, one per
        distinct file referenced across `script_paths`.
    """
    commands = []
    seen_filenames: Set[str] = set()
    for script_path in script_paths:
        text = script_path.read_text()
        header, blocks, _ = split_curl_script(text)
        folder_match = FOLDER_RE.search(header)
        download_folder = (
            folder_match.group(1).strip() if folder_match else "downloads"
        )
        for filename, block in blocks:
            if filename in seen_filenames:
                continue
            curl_line = next(
                (
                    line
                    for line in block.splitlines()
                    if line.startswith("curl ")
                ),
                None,
            )
            if curl_line is not None:
                seen_filenames.add(filename)
                commands.append((filename, curl_line, download_folder))
    return commands


def next_try_number(script_dir: Path, name: Optional[str] = None) -> int:
    """The next unused try number in `script_dir`.

    Looks for ``{n}.sh`` files, or ``{name}_{n}.sh`` if `name` is
    given.
    """
    if name:
        pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.sh$")
        glob_pattern = f"{name}_*.sh"
    else:
        pattern = re.compile(r"^(\d+)\.sh$")
        glob_pattern = "*.sh"
    numbers = [
        int(match.group(1))
        for f in glob.glob(str(script_dir / glob_pattern))
        if (match := pattern.match(os.path.basename(f)))
    ]
    return max(numbers, default=0) + 1


def combine_scripts(
    script_paths: List[Path], existing_filenames: Set[str]
) -> Tuple[Optional[str], int, int]:
    """Merge `script_paths` into one filtered curl script.

    Files referenced by more than one input script (e.g. overlapping
    curl scripts from separate MAST queries) are only included once.

    Returns
    -------
    `tuple`
        ``(combined_text, n_remaining, n_unique)``, where `n_unique`
        is the number of distinct files referenced across
        `script_paths`. `combined_text` is `None` if every unique file
        is already downloaded.
    """
    header: Optional[str] = None
    footer = ""
    unique_blocks: List[Tuple[Optional[str], str]] = []
    seen_filenames: Set[str] = set()
    for script_path in script_paths:
        block_header, blocks, block_footer = split_curl_script(
            script_path.read_text()
        )
        if header is None:
            header = block_header
        footer = block_footer
        for filename, block in blocks:
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)
            unique_blocks.append((filename, block))

    remaining = [
        (filename, block)
        for filename, block in unique_blocks
        if filename not in existing_filenames
    ]
    if not remaining:
        return None, 0, len(unique_blocks)
    combined = (
        (header or "") + "".join(block for _, block in remaining) + footer
    )
    return combined, len(remaining), len(unique_blocks)


def write_resume_script(
    script_paths: List[Path],
    name: Optional[str] = None,
    later_stage_base_dir: Optional[Path] = None,
    input_crds: int = 1584,
    cache: Optional[Dict[Tuple[str, str, int], bool]] = None,
) -> Optional[Path]:
    """Filter and merge `script_paths`, writing one resume script.

    Parameters
    ----------
    script_paths : `list` of `Path`
        MAST bundle curl scripts to filter and merge.
    name : `str`, optional
        Base name for the output script, written as ``{name}_{n}.sh``.
        If omitted, the script is written as bare ``{n}.sh``.
    later_stage_base_dir : `Path`, optional
        Forwarded to `existing_uncal_filenames` so a file whose RATE,
        CAL, or stage 3 output already exists is also excluded, even
        if its UNCAL file isn't present locally. Default is `None`
        (UNCAL completeness only).
    input_crds : `int`, optional
        Forwarded to `existing_uncal_filenames`. Default is 1584.
    cache : `dict`, optional
        Forwarded to `existing_uncal_filenames`/`later_stage_output_exists`.
        Default is `None` (no caching).

    Returns
    -------
    `Path` or `None`
        Path to the (possibly pre-existing, see above) resume script,
        or `None` if every file referenced by `script_paths` is
        already downloaded.
    """
    downloads_dir = script_paths[0].resolve().parent
    existing_filenames = existing_uncal_filenames(
        downloads_dir,
        later_stage_base_dir=later_stage_base_dir,
        input_crds=input_crds,
        cache=cache,
    )
    try_n = next_try_number(downloads_dir, name)

    combined, n_remaining, n_total = combine_scripts(
        script_paths, existing_filenames
    )
    galfind_logger.info(
        f"{n_total - n_remaining}/{n_total} files already downloaded, "
        f"{n_remaining} remaining"
    )
    if combined is None:
        return None

    # reuse the most recent existing resume script if it already covers
    # this exact set of remaining files, instead of writing a redundant
    # near-duplicate (e.g. if this is called twice in a row with no
    # downloads completing in between)
    if try_n > 1:
        prev_name = f"{name}_{try_n - 1}.sh" if name else f"{try_n - 1}.sh"
        prev_path = downloads_dir / prev_name
        if prev_path.exists():
            _, new_blocks, _ = split_curl_script(combined)
            _, prev_blocks, _ = split_curl_script(prev_path.read_text())
            if {fn for fn, _ in new_blocks} == {fn for fn, _ in prev_blocks}:
                galfind_logger.info(
                    f"{prev_path} already covers these {n_remaining} "
                    "remaining file(s); reusing it instead of writing "
                    "a new resume script"
                )
                return prev_path

    out_name = f"{name}_{try_n}.sh" if name else f"{try_n}.sh"
    out_path = downloads_dir / out_name
    out_path.write_text(combined)
    out_path.chmod(0o755)
    return out_path


class Raw_JWST_Data:
    """Downloads and reduces raw JWST NIRCam imaging data through the
        JWST calibration pipeline.

    Wraps `astroquery` MAST queries/downloads and the ``jwst``
    pipeline's Stage 1-3 processing (detector-level calibration, image
    calibration, and association-based resampling/combination) for a
    single JWST programme ID.

    Parameters
    ----------
    survey : `str`
        Name of the survey this data belongs to.
    pid : `int`
        JWST proposal/programme ID to query and reduce data for.
    instrument : `Type[Instrument]`, optional
        The `Instrument` subclass to reduce data for. Only `NIRCam` is
        currently supported. Default is `NIRCam`.

    Attributes
    ----------
    instrument : `Instrument`
        Instance of the `instrument` class.
    survey : `str`
        Name of the survey.
    pid : `int`
        JWST proposal/programme ID.
    download_products : `list` of `str`
        Local paths to the UNCAL products' download scripts. Only set
        once `query_mast` has been called.
    """

    def __init__(
        self: Self,
        survey: str,
        pid: int,
        instrument: Type[Instrument] = NIRCam,
    ):
        if instrument.__name__ != "NIRCam":
            raise InvalidOptionError(
                f"instrument={instrument.__name__!r} not supported; "
                "Raw_JWST_Data currently only supports 'NIRCam'."
            )
        self.instrument = instrument()
        self.survey = survey
        self.pid = pid
        # shared across `query_mast`/`download`/`move_uncals` within
        # this instance's lifetime, so a file's RATE/CAL/science
        # completeness (expensive - opens each file) isn't
        # re-verified by every method that needs to know, when
        # nothing on disk could have changed the answer in between
        self._later_stage_cache: Dict[Tuple[str, str, int], bool] = {}
        # accumulated across every `run_stage3` call made on this
        # instance (e.g. one per `asn_file_search_dir` in a loop), so
        # the same association isn't re-read from disk every time, and
        # so every i2d output produced this session stays inspectable
        # afterwards - keyed by output_dir, then by association file;
        # see the module-level `stage3_output_paths` function
        self.stage3_output_paths: Dict[str, Dict[str, Path]] = {}

    @property
    def folder_name(self: Self) -> str:
        """`str`: Local directory this programme's raw/reduced data is
        stored under.

        `"{GALFIND_DATA}/{facility_name}/PID={pid}"`.
        """
        base_data = config["DEFAULT"]["GALFIND_DATA"]
        facility = self.instrument.facility.__class__.__name__.lower()
        return f"{base_data}/{facility}/PID={self.pid}"

    def __repr__(self: Self) -> str:
        class_name = self.instrument.__class__.__name__
        return f"Raw_{class_name}_Data({self.survey},PID={self.pid})"

    def __str__(self: Self) -> str:
        class_name = self.instrument.__class__.__name__
        summary = f"{class_name} data for {self.survey} (PID={self.pid})"
        n_files = getattr(self, "n_files", None)
        n_uncals = getattr(self, "n_uncals", None)
        if n_files is not None and n_uncals is not None:
            summary += f": {n_uncals}/{n_files} UNCAL files downloaded"
        return summary

    def __call__(
        self: Self,
        split_by: str = "sky",
        n_cores: int = 1,
        input_crds: int = 1584,
        pre_download_refs: bool = False,
        make_asn_kwargs: Dict[str, Any] = {},
        # stage1_steps: Dict[str, Any] = {},
        # stage2_steps: Dict[str, Any] = {},
        # stage3_steps: Dict[str, Any] = {},
    ) -> List[Data]:
        """Run the full raw-to-reduced pipeline: download through stage 3.

        Parameters
        ----------
        split_by : `str`, optional
            How to group exposures into stage 3 associations. Forwarded
            to `make_asn` as `split_by`. Default is ``"sky"`` (group by
            pointing).
        n_cores : `int`, optional
            Number of CPU cores to use for parallel processing. Forwarded
            to `download`, `run_stage1`, `run_stage2`, and `run_stage3`.
            Default is 1 (no parallelism).
        input_crds : `int`, optional
            CRDS context version to use for the JWST pipeline. Forwarded
            to `download`, `run_stage1`, `run_stage2`, and `run_stage3`.
            Default is 1584.
        pre_download_refs : `bool`, optional
            Whether to pre-download reference files for the JWST pipeline
            before running each stage. Forwarded to `download`, `run_stage1`,
            `run_stage2`, and `run_stage3`. Default is `False`.
        make_asn_kwargs : `dict`, optional
            Extra keyword arguments forwarded to `make_asn` (e.g.
            `hdr_cols`, `match_radius`, `plot`); `split_by` is set
            from `subdivide` and `input_crds` from `input_crds`, so
            neither should be given here. Default is empty dict.

        Returns
        -------
        `list` of `Data`
            One `Data` object per survey (pointing) associations were
            generated for, built from that pointing's freshly-reduced
            stage 3 imaging.
        """
        if self.instrument.__class__.__name__ == "NIRCam":
            return self._call_nircam(
                split_by=split_by,
                n_cores=n_cores,
                input_crds=input_crds,
                pre_download_refs=pre_download_refs,
                make_asn_kwargs=make_asn_kwargs,
                # stage1_steps=stage1_steps,
                # stage2_steps=stage2_steps,
                # stage3_steps=stage3_steps,
            )
        else:
            raise NotImplementedError(
                f"{self.instrument.__class__.__name__} is not implemented "
                "for Raw_JWST_Data.__call__"
            )

    def _call_nircam(
        self: Self,
        split_by: str = "sky",
        n_cores: int = 1,
        input_crds: int = 1584,
        pre_download_refs: bool = False,
        make_asn_kwargs: Dict[str, Any] = {},
        surveys: Optional[List[str]] = None,
        # stage3_steps: Dict[str, Any] = {},
    ) -> List[Data]:
        import jwst
        from snowblind import JumpPlusStep, SnowblindStep

        # download the data from MAST
        self.download(n_cores=n_cores)
        self.move_uncals()

        # run the stage 1 pipeline
        stage1_steps = {
            "jump": {
                "expand_large_events": False,
                "post_hooks": [
                    f"{SnowblindStep.__module__}.{SnowblindStep.__qualname__}",
                    f"{JumpPlusStep.__module__}.{JumpPlusStep.__qualname__}",
                ],
            },
            "clean_flicker_noise": {
                "skip": False,
            },
        }
        self.run_stage1(
            input_crds=input_crds,
            steps=stage1_steps,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
        )

        # run stage 2 of the JWST pipeline
        stage2_steps = {}
        self.run_stage2(
            input_crds=input_crds,
            steps=stage2_steps,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
        )
        # run post stage 2 steps - bg subtraction and wisp removal

        # generate stage 3 associations - one group (pointing) per
        # returned asn_file_search_dir
        asn_file_search_dirs = self.make_asn(
            split_by=split_by,
            input_crds=input_crds,
            **make_asn_kwargs,
        )
        if surveys is None:
            surveys = [
                Path(d).name for d in asn_file_search_dirs if Path(d).is_dir()
            ]

        # run the stage 3 pipeline for each association group
        stage3_steps = {
            "tweakreg": {
                "skip": False,
                "abs_refcat": "GAIADR3",
                "abs_minobj": 3,
                "save_abs_catalog": True,
            }
        }
        for survey in surveys:
            self.run_stage3(
                input_crds=input_crds,
                steps=stage3_steps,
                asn_file_search_dir=surveys,
                n_cores=n_cores,
                pre_download_refs=pre_download_refs,
            )

        # alias identifying this exact stage1/stage2/stage3
        # configuration, jwst pipeline version, and CRDS/PMAP context -
        # the same configuration always resolves to the same alias; a
        # genuinely new configuration is auto-registered under the
        # next unused "vN" in configs/reduction_versions.json
        version = get_or_register_reduction_version(
            reduction_version_key(
                stage1_steps,
                stage2_steps,
                stage3_steps,
                jwst.__version__,
                f"jwst_{input_crds}.pmap",
            )
        )
        # symlink stage 3 i2d outputs into the survey/version/pixel-
        # scale layout Data.pipeline expects, then build a Data object
        # for every survey (pointing) touched
        symlinks_by_survey = self.ingest_stage3_outputs(version=version)
        instrument_names = [self.instrument.__class__.__name__]
        return [
            Data.from_survey_version_psfs(
                survey,
                version,
                instrument_names=instrument_names,
                # pix_scales={"NIRCam": 0.063 * u.arcsec},
                psfs=None,
            )
            for survey in tqdm(
                symlinks_by_survey,
                desc="Building Data objects",
                total=len(symlinks_by_survey),
            )
        ]

    @run_in_self_dir(lambda self: f"{self.folder_name}/downloads")
    @log_time(logging.INFO, u.min)
    def query_mast(
        self: Self,
        input_crds: int = 1584,
        skip_if_processed: bool = True,
    ) -> List[str]:
        """Query MAST for JWST uncalibrated data products.

        Downloads uncalibrated (UNCAL) raw data from the MAST archive for the
        specified program ID and instrument.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version whose RATE/CAL/science output
            directories are checked for already-processed files when
            `skip_if_processed` is `True`. Default is 1584.
        skip_if_processed : `bool`, optional
            If `True` (default), an UNCAL file that isn't downloaded
            locally but whose RATE, CAL, or stage 3 output already
            exists (e.g. because it was intentionally removed by
            `remove_uncals_with_rate` once no longer needed) is not
            queued for (re)download. Set to `False` to force queuing
            every UNCAL file MAST expects, regardless of downstream
            progress.

        Returns
        -------
        `list` of `str`
            Local file paths to downloaded UNCAL data products.
        """
        instrument_name = f"{self.instrument.__class__.__name__.upper()}/IMAGE"
        later_stage_base_dir = (
            Path(self.folder_name) if skip_if_processed else None
        )

        # if curl scripts already exist locally (either the original
        # MAST-generated ones or a previous resume script), filter them
        # down to the still-missing files instead of re-querying MAST
        # for a fresh set. The original scripts are preferred as the
        # source (over the narrower numbered resume chain) since they
        # cover the full, unchanging product list: a numbered script
        # only carries forward whatever was missing at the time it was
        # generated, so a file that later becomes incomplete again
        # (e.g. overwritten by a failed retry) can silently fall out of
        # that chain and never get re-queued if it's used as the source
        original_scripts = sorted(glob.glob("mastDownload_*.sh"))
        if original_scripts:
            source_scripts = original_scripts
        else:
            numbered_scripts = [
                f
                for f in glob.glob("*.sh")
                if os.path.splitext(os.path.basename(f))[0].isdigit()
            ]
            source_scripts = (
                [max(numbered_scripts, key=lambda f: int(Path(f).stem))]
                if numbered_scripts
                else []
            )
        if source_scripts:
            galfind_logger.info(
                f"Found existing curl script(s) {source_scripts} for "
                f"{instrument_name=} {self.pid=}; filtering locally "
                "instead of re-querying MAST"
            )
            out_path = write_resume_script(
                [Path(p) for p in source_scripts],
                later_stage_base_dir=later_stage_base_dir,
                input_crds=input_crds,
                cache=self._later_stage_cache,
            )
            self.download_products = (
                [str(out_path)] if out_path is not None else []
            )
            return self.download_products

        from astropy.table import Table
        from astroquery.mast import Observations

        save_path = (
            f"{self.instrument.__class__.__name__}_{self.pid}_uncals.fits"
        )

        if os.path.isfile(save_path):
            filtered_data_products = Table.read(save_path)
            # re-check even a cached manifest, in case it was saved by
            # an older version of this method that didn't filter by
            # detector suffix (see the fresh-query branch below)
            filtered_data_products = filtered_data_products[
                [
                    bool(NIRCAM_DETECTOR_RE.search(filename))
                    for filename in filtered_data_products["productFilename"]
                ]
            ]
            galfind_logger.info(
                f"{save_path} already exists with "
                f"{len(filtered_data_products)} entries; skipping MAST "
                "query stage"
            )
        else:
            obs_table = Observations.query_criteria(
                instrument_name=instrument_name,
                proposal_id=str(self.pid),
            )
            # print(obs_table, obs_table["target_name"], obs_table.colnames)
            data_products = Observations.get_product_list(obs_table)
            # save product list
            filtered_data_products = Observations.filter_products(
                data_products,
                productSubGroupDescription="UNCAL",
            )
            # belt-and-braces: the observation-level `instrument_name`
            # query above should already restrict to NIRCam imaging,
            # but products aren't re-tagged with instrument, so also
            # check each filename's detector suffix directly rather
            # than trusting that filter to have propagated correctly
            filtered_data_products = filtered_data_products[
                [
                    bool(NIRCAM_DETECTOR_RE.search(filename))
                    for filename in filtered_data_products["productFilename"]
                ]
            ]

            if (
                filtered_data_products is None
                or len(filtered_data_products) == 0
            ):
                galfind_logger.warning(
                    f"No UNCAL products found for {instrument_name=} "
                    f"{self.pid=} on MAST; skipping download"
                )
                self.download_products = []
                return self.download_products

            filtered_data_products.write(save_path, overwrite=True)
            galfind_logger.info(
                f"{len(filtered_data_products)} entries saved to {save_path}"
            )

        # only queue UNCAL files that are not already fully downloaded,
        # either still nested under downloads/ or already moved to uncals/
        existing_filenames = existing_uncal_filenames(
            Path.cwd(),
            later_stage_base_dir=later_stage_base_dir,
            input_crds=input_crds,
            cache=self._later_stage_cache,
        )
        n_expected = len(filtered_data_products)
        missing_data_products = filtered_data_products[
            [
                os.path.basename(filename) not in existing_filenames
                for filename in filtered_data_products["productFilename"]
            ]
        ]
        n_missing = len(missing_data_products)
        if n_missing == 0:
            galfind_logger.info(
                f"All {n_expected} UNCAL files already downloaded for "
                f"{instrument_name=} {self.pid=}; skipping MAST download "
                "stage (curl script generation + download)"
            )
            self.download_products = []
            return self.download_products
        galfind_logger.info(
            f"{n_expected - n_missing}/{n_expected} UNCAL files already "
            f"downloaded for {instrument_name=} {self.pid=}; queuing the "
            f"remaining {n_missing} file(s) for download"
        )

        # MAST's bundle endpoint rejects requests with more than 1000
        # fields (i.e. products), so download in chunks
        mast_bundle_chunk_size = 1000
        manifests = [
            Observations.download_products(
                missing_data_products[i : i + mast_bundle_chunk_size],
                curl_flag=True,
                verbose=False,
            )
            for i in range(0, n_missing, mast_bundle_chunk_size)
        ]
        manifest = vstack(manifests)
        self.download_products = manifest["Local Path"].tolist()
        galfind_logger.info(
            f"Queried {self.download_products=} products for "
            f"{instrument_name=} {self.pid=} from MAST"
        )
        return self.download_products

    @run_in_self_dir(lambda self: f"{self.folder_name}/downloads")
    @log_time(logging.INFO, u.hour)
    def download(
        self: Self,
        n_cores: int = 1,
        input_crds: int = 1584,
        skip_if_processed: bool = True,
    ) -> None:
        """Download MAST data products using curl scripts.

        Runs the individual per-file curl commands extracted from the
        curl script(s) in `self.download_products`, so progress can be
        tracked at the file level regardless of how many curl scripts
        `self.download_products` contains. Sets `self.n_files` to the
        total number of UNCAL files MAST expects for this programme
        (per the saved product manifest), regardless of how many are
        actually present once this completes.

        Parameters
        ----------
        n_cores : `int`, optional
            Number of files to download concurrently. Default is 1
            (sequential). Downloads are network-bound, so this is run
            with threads rather than separate processes.
        input_crds : `int`, optional
            CRDS context version whose RATE/CAL/science output
            directories are checked for already-processed files when
            `skip_if_processed` is `True`. Default is 1584.
        skip_if_processed : `bool`, optional
            If `True` (default), an UNCAL file that isn't present
            locally but whose RATE, CAL, or stage 3 output already
            exists (e.g. because it was intentionally removed by
            `remove_uncals_with_rate` once no longer needed) is not
            re-downloaded. Set to `False` to force downloading every
            UNCAL file MAST expects, regardless of downstream
            progress.
        """
        # always re-check what's missing rather than reusing a
        # previous call's result: `query_mast` is cheap once local
        # scripts exist (no MAST calls), and a caller (e.g.
        # `move_uncals`) retrying `download` needs an up-to-date view,
        # not a stale one left over from an earlier, already-run call
        self.query_mast(
            input_crds=input_crds, skip_if_processed=skip_if_processed
        )
        self.n_files = len(expected_uncal_sizes(Path.cwd()))
        if not self.download_products:
            galfind_logger.info(
                "No curl scripts to run; skipping download stage"
            )
            return
        # captured once, up front: `os.chdir` (via `run_in_self_dir`) is
        # process-global, not thread-local, so relying on the ambient
        # cwd inside concurrently-running curl subprocesses is unsafe -
        # anything else in this process that changes directory while
        # workers are still in flight would misdirect them. Pinning an
        # explicit `cwd` per subprocess below removes that dependency
        # entirely.
        downloads_dir = os.getcwd()
        commands = extract_curl_commands(
            [Path(p) for p in self.download_products]
        )
        n_already_downloaded = self.n_files - len(commands)
        galfind_logger.info(
            f"Downloading {len(commands)} "
            f"{self.instrument.__class__.__name__} PID={self.pid} UNCAL "
            "file(s); this may take a while for programmes with many files"
        )

        def _run_curl(curl_command: str, download_folder: str) -> None:
            process = subprocess.Popen(
                ["bash", "-c", curl_command],
                cwd=downloads_dir,
                env={**os.environ, "DOWNLOAD_FOLDER": download_folder},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.wait()

        desc = (
            f"Downloading {self.instrument.__class__.__name__} "
            f"PID={self.pid} products"
        )
        if n_cores > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                futures = [
                    executor.submit(_run_curl, curl_command, download_folder)
                    for _, curl_command, download_folder in commands
                ]
                for future in tqdm(
                    as_completed(futures),
                    desc=desc,
                    initial=n_already_downloaded,
                    total=self.n_files,
                    unit="file",
                ):
                    future.result()
        else:
            for _, curl_command, download_folder in tqdm(
                commands,
                desc=desc,
                initial=n_already_downloaded,
                total=self.n_files,
                unit="file",
            ):
                _run_curl(curl_command, download_folder)

    @run_in_self_dir(lambda self: f"{self.folder_name}/downloads")
    def write_resume_script(self: Self) -> Optional[str]:
        """Write a single curl script that resumes a halted MAST download.

        Merges the curl scripts already generated for this programme
        (`self.download_products`) and filters them down to only the
        UNCAL files not yet present on disk, writing the result as one
        runnable script named ``{n}.sh`` alongside the originals, with
        ``n`` incrementing on each call so repeated resumes keep
        shrinking. Use in place of `download` after a run was
        interrupted, to avoid re-querying MAST for files that have
        already been downloaded.

        Returns
        -------
        `str` or `None`
            Local path to the newly-written resume script, or `None`
            if every file referenced by `self.download_products` is
            already downloaded.
        """
        if not getattr(self, "download_products", None):
            galfind_logger.info(
                f"No curl scripts recorded for {self.pid=}; nothing to "
                "resume"
            )
            return None

        script_paths = [Path(p) for p in self.download_products]
        out_path = write_resume_script(script_paths)
        if out_path is None:
            galfind_logger.info(
                f"All files already downloaded for {self.pid=}; no "
                "resume script written"
            )
            return None
        galfind_logger.info(f"Wrote resume script for {self.pid=}: {out_path}")
        return str(out_path)

    @run_in_self_dir(lambda self: self.folder_name)
    def move_uncals(
        self: Self,
        input_crds: int = 1584,
        skip_if_processed: bool = True,
    ) -> None:
        """Organize uncalibrated data files into a dedicated directory.

        Before moving anything, verifies every UNCAL file MAST expects
        for this programme has been fully downloaded (per the saved
        product manifest's expected file sizes). If any are missing or
        were left truncated by an interrupted download, warns and
        downloads them before proceeding.

        Moves all UNCAL FITS files from nested download directories
        into a single 'uncals' directory for easier access. Sets
        `self.n_uncals` to the number of UNCAL files present under
        ``uncals/`` once this completes.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version whose RATE/CAL/science output
            directories are checked for already-processed files when
            `skip_if_processed` is `True`. Default is 1584.
        skip_if_processed : `bool`, optional
            If `True` (default), an UNCAL file that isn't present
            locally but whose RATE, CAL, or stage 3 output already
            exists (e.g. because it was intentionally removed by
            `remove_uncals_with_rate` once no longer needed) is not
            treated as missing, so it doesn't trigger a `download`
            call. Set to `False` to force downloading every UNCAL
            file MAST expects, regardless of downstream progress.
        """
        downloads_dir = Path.cwd() / "downloads"
        expected_sizes = expected_uncal_sizes(downloads_dir)
        if expected_sizes:
            # `Path(self.folder_name)`, not `Path.cwd()`: both resolve
            # to the same directory here (this method runs inside it
            # via `run_in_self_dir`), but matching `query_mast`'s exact
            # string means the two share cache hits in
            # `self._later_stage_cache` instead of missing on a
            # symlink-resolution difference between the two spellings
            later_stage_base_dir = (
                Path(self.folder_name) if skip_if_processed else None
            )
            complete_filenames = existing_uncal_filenames(
                downloads_dir,
                later_stage_base_dir=later_stage_base_dir,
                input_crds=input_crds,
                cache=self._later_stage_cache,
            )
            missing_filenames = set(expected_sizes) - complete_filenames
            if missing_filenames:
                galfind_logger.warning(
                    f"{len(missing_filenames)}/{len(expected_sizes)} "
                    f"UNCAL file(s) not fully downloaded for "
                    f"{self.pid=}; downloading before moving to "
                    "'uncals'"
                )
                self.download(
                    input_crds=input_crds,
                    skip_if_processed=skip_if_processed,
                )

        uncals = glob.glob("downloads/*/*/*/*_uncal.fits")
        # move all of these files to an uncals directory, skipping any
        # that still don't match their expected size (e.g. left behind
        # by a download that failed silently)
        os.makedirs("uncals", exist_ok=True)
        for file in uncals:
            filename = os.path.basename(file)
            expected_size = expected_sizes.get(filename)
            actual_size = os.path.getsize(file)
            if expected_size is not None and actual_size != expected_size:
                galfind_logger.warning(
                    f"Not moving {filename!r}: expected {expected_size} "
                    f"bytes but found {actual_size}; still incomplete"
                )
                continue
            os.rename(file, f"uncals/{filename}")

        self.n_uncals = sum(
            1
            for f in glob.glob("uncals/*_uncal.fits")
            if expected_sizes.get(os.path.basename(f))
            in (None, os.path.getsize(f))
        )

    @staticmethod
    def remove_if_next_stage_exists(
        filename: str,
        output_dir: str,
        output_suffix: str,
        dry_run: bool = True,
    ) -> bool:
        """Delete `filename` if the next stage's output for it already exists.

        Uses `stage_output_path` to determine where the next pipeline
        stage would (or did) write its output for `filename`, and only
        removes `filename` once that output is confirmed complete on
        disk (via `is_complete_pipeline_output`) - i.e. once `filename`
        is no longer needed to reproduce it. A RATE/CAL file left
        truncated by an interrupted run is therefore *not* treated as
        "already produced", so its UNCAL input is kept rather than
        deleted out from under a retry.

        Parameters
        ----------
        filename : `str`
            Path to the input file (e.g. an UNCAL FITS file) to
            conditionally remove.
        output_dir : `str`
            Directory the next stage writes its output into (e.g.
            ``"rate_1584"``).
        output_suffix : `str`
            Filename suffix the next stage uses for its output (e.g.
            ``"rate"``).
        dry_run : `bool`, optional
            If `True` (default), only logs what would be removed
            without deleting anything. Must be explicitly set to
            `False` to actually delete `filename`.

        Returns
        -------
        `bool`
            Whether `filename` was (or, in a dry run, would be)
            removed.
        """
        out_path = stage_output_path(filename, output_dir, output_suffix)
        in_path = Path(filename)
        if not is_complete_pipeline_output(out_path):
            galfind_logger.debug(
                f"Not removing {filename!r}: expected next-stage output "
                f"{str(out_path)!r} does not exist yet or is incomplete"
            )
            return False
        if not in_path.is_file():
            galfind_logger.debug(f"{filename!r} already removed")
            return False
        if dry_run:
            galfind_logger.info(
                f"[dry_run] Would remove {filename!r}: "
                f"{str(out_path)!r} already exists"
            )
            return True
        galfind_logger.warning(
            f"Removing {filename!r}: {str(out_path)!r} already exists"
        )
        in_path.unlink()
        return True

    @run_in_self_dir(lambda self: self.folder_name)
    def remove_uncals_with_rate(
        self: Self,
        input_crds: int = 1584,
        dry_run: bool = True,
    ) -> List[str]:
        """Delete UNCAL files whose corresponding RATE file already exists.

        Frees disk space once stage 1 (`run_stage1`) has produced a
        RATE file for a given UNCAL file, since it is the RATE file -
        not the raw UNCAL - that stage 2 (`run_stage2`) consumes.
        Delegates the per-file check/delete to
        `remove_if_next_stage_exists`, so nothing is removed unless its
        RATE output is already on disk.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS pipeline version number matching the RATE files'
            output directory (``rate_{input_crds}``). Default is 1584.
        dry_run : `bool`, optional
            If `True` (default), only logs which UNCAL files would be
            removed without deleting anything. Must be explicitly set
            to `False` to actually delete files.

        Sets `self.n_uncals` (total UNCAL file count) if not already
        set elsewhere (e.g. by `move_uncals`), and always (re)sets
        `self.n_uncals_with_rate` to the number of those UNCAL files
        that already have a RATE output, logging both counts.

        Returns
        -------
        `list` of `str`
            UNCAL filenames that were (or, in a dry run, would be)
            removed.
        """
        output_dir = f"rate_{input_crds}"
        uncal_files = sorted(glob.glob("uncals/*_uncal.fits"))
        if not hasattr(self, "n_uncals"):
            self.n_uncals = len(uncal_files)
        removed = [
            filename
            for filename in uncal_files
            if self.remove_if_next_stage_exists(
                filename, output_dir, "rate", dry_run=dry_run
            )
        ]
        self.n_uncals_with_rate = len(removed)
        galfind_logger.info(
            f"{self.n_uncals_with_rate}/{self.n_uncals} UNCAL file(s) "
            f"for {self.pid=} have an existing RATE output"
        )
        verb = "Would remove" if dry_run else "Removed"
        galfind_logger.info(
            f"{verb} {len(removed)}/{len(uncal_files)} UNCAL file(s) "
            f"with an existing RATE output for {self.pid=}"
        )
        return removed

    @staticmethod
    def set_crds_context(input_crds: int = 1584) -> str:
        """Set the CRDS context for JWST calibration reference data.

        Configures the CRDS (Calibration Reference Data System) context file
        for JWST data reduction pipeline.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS pipeline version number. Default is 1584.

        Returns
        -------
        `str`
            CRDS context file name (e.g., "jwst_1584.pmap").
        """
        import crds

        crds_context = f"jwst_{input_crds}.pmap"
        try:
            crds.client.get_reference_names(crds_context)
            galfind_logger.debug(
                f"{crds_context=} is valid and all files are accessible."
            )
            os.environ["CRDS_CONTEXT"] = crds_context
            galfind_logger.info(f"Set {crds_context=} for JWST data reduction")
        except crds.exceptions.CrdsError as e:
            raise ExternalToolError(
                f"crds_context={crds_context!r} failed certification: {e}"
            ) from e
        return crds_context

    @staticmethod
    def pre_download_refs(
        filenames: Union[NDArray[str], List[str]],
        input_crds: int = 1584,
    ) -> List[str]:
        """Pre-download CRDS calibration reference files for JWST data.

        Retrieves all calibration reference files needed for the specified
        UNCAL data files before running the full reduction pipeline.

        Parameters
        ----------
        filenames : `numpy.ndarray` or `list` of `str`
            Paths to UNCAL FITS files requiring calibration references.
        input_crds : `int`, optional
            CRDS pipeline version number. Default is 1584.
        """
        import crds

        os.environ["CRDS_PATH"] = (
            f"{config['DEFAULT']['GALFIND_DATA']}/crds_cache"
        )
        crds_context = Raw_JWST_Data.set_crds_context(input_crds)
        suffixes = np.unique(
            [file.split("_")[-1].replace(".fits", "") for file in filenames]
        )
        if len(suffixes) != 1:
            raise GalfindError(
                "Expected all files to have the same suffix, but found "
                f"suffixes={suffixes!r}."
            )
        suffix = suffixes[0]
        [
            crds.getreferences(
                dict(fits.getheader(file)), context=crds_context
            )
            for file in tqdm(
                filenames,
                desc=(
                    f"Downloading CRDS references to "
                    f"{os.environ['CRDS_PATH']} for {suffix} files with "
                    f"{os.environ['CRDS_CONTEXT']=}"
                ),
                total=len(filenames),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]

    @run_in_self_dir(lambda self: self.folder_name)
    @log_time(logging.INFO, u.s)
    def make_asn(
        self: Self,
        split_by: Optional[str, List[str]] = None,
        input_crds: int = 1584,
        hdr_cols: List[str] = [
            "TARG_RA",
            "TARG_DEC",
            "FILTER",
            "OBSERVTN",
            "PROGRAM",
            "TARGPROP",
            "OBS_ID",
        ],
        match_radius: u.Quantity = 25.0 * u.arcmin,
        plot: bool = True,
    ) -> List[str]:
        """Create association (ASN) tables for JWST pipeline processing.

        Groups calibrated data files into associations based on sky position,
        filter, or other header criteria for processing through the JWST
        reduction pipeline.

        Parameters
        ----------
        split_by : `str` or `list` of `str`, optional
            Grouping criterion ("sky" to group by position). Default is None.
        input_crds : `int`, optional
            CRDS pipeline version. Default is 1584.
        hdr_cols : `list` of `str`, optional
            FITS header columns to track. Default includes RA, DEC,
            FILTER, etc.
        match_radius : `astropy.units.Quantity`, optional
            Sky position matching radius. Default is 25 arcmin.
        plot : `bool`, optional
            Whether to generate diagnostic plots. Default is True.

        Returns
        -------
        `list` of `str`
            Each group ID (pointing) associations were generated for -
            usable directly as `run_stage3`'s `asn_file_search_dir`.
        """
        # set CRDS context
        self.set_crds_context(input_crds)
        cal_filenames = np.array(glob.glob(f"cal_{input_crds}/*_cal.fits"))
        # populate dictionary with relevant header info
        hdr_info = {
            colname: np.full(len(cal_filenames), None, dtype=object)
            for colname in hdr_cols + ["CRVAL1", "CRVAL2"]
        }
        for i, filename in tqdm(
            enumerate(cal_filenames),
            desc=f"Reading {repr(self)} headers",
            total=len(cal_filenames),
        ):
            with fits.open(filename) as hdul:
                hdr = hdul[0].header
                for colname in hdr_cols:
                    hdr_info[colname][i] = hdr.get(colname, "UNKNOWN")
                sci_hdr = hdul["SCI"].header
                for colname in ["CRVAL1", "CRVAL2"]:
                    hdr_info[colname][i] = sci_hdr.get(colname, "UNKNOWN")

        # perform appropriate split
        if split_by is not None:
            if split_by == "sky":
                sky_coords = SkyCoord(
                    np.array(hdr_info["CRVAL1"]).astype(float) * u.deg,
                    np.array(hdr_info["CRVAL2"]).astype(float) * u.deg,
                )
                groups = funcs.group_positions(
                    sky_coords, match_radius=match_radius
                )
                groups = {
                    f"{self.survey}-{name}": filenames
                    for name, filenames in groups.items()
                }
                plot_subdir = (
                    f"sky<{match_radius.to(u.arcmin).value:.1f}arcmin"
                )
            else:
                raise InvalidOptionError(
                    f"split_by={split_by!r} not in ['sky']."
                )
        else:
            groups = {self.survey: cal_filenames}

        if plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            # plot all footprints
            all_footprints = funcs.footprints_from_files(cal_filenames)
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("Dec [deg]")
            plt.grid(True)
            ax.invert_xaxis()  # RA increases to the left

            for f, coords in all_footprints.items():
                poly = Polygon(
                    coords,
                    closed=False,
                    fill=True,
                    alpha=0.3,
                    facecolor="grey",
                    edgecolor="k",
                )
                ax.add_patch(poly)

            for group_id, group in groups.items():
                galfind_logger.info(f"Group {group_id} has {len(group)} files")
                footprints = funcs.footprints_from_files(cal_filenames[group])
                added_poly = []
                for f, coords in footprints.items():
                    poly = Polygon(
                        coords,
                        closed=False,
                        fill=True,
                        alpha=0.75,
                        facecolor="green",
                        edgecolor="k",
                    )
                    ax.add_patch(poly)
                    added_poly.append(poly)
                margin = 1.0
                all_coords = np.vstack([poly.get_xy() for poly in added_poly])
                xmin, ymin = all_coords.min(axis=0)
                xmax, ymax = all_coords.max(axis=0)
                dx = (xmax - xmin) * margin
                dy = (ymax - ymin) * margin
                ax.set_xlim(xmin - dx, xmax + dx)
                ax.set_ylim(ymin - dy, ymax + dy)

                plot_dir = f"{self.folder_name}/asn_{input_crds}/{plot_subdir}"
                save_path = f"{plot_dir}/{group_id}.png"
                funcs.make_dirs(save_path)
                plt.savefig(save_path)
                # remove all patches from the axes
                for poly in added_poly:
                    poly.remove()
            plt.close()

        # Generate with explicit ruleset that groups by filter
        for group_id, group in tqdm(
            groups.items(),
            desc=f"Generating associations for {self.survey} {self.pid=}",
            total=len(groups),
            disable=galfind_logger.getEffectiveLevel() > logging.INFO,
        ):
            # split by filter
            all_filt_names = np.unique(hdr_info["FILTER"][group])
            for filt in all_filt_names:
                # galfind_logger.info(
                #     f"Generating association for {group_id} with "
                #     f"{len(group)} files"
                # )
                product_name = f"{group_id}-{filt}"
                filt_group = np.array(
                    [id for id in group if hdr_info["FILTER"][id] == filt]
                )
                product_filenames = cal_filenames[filt_group]
                product_subdir = f"asn_{input_crds}/{group_id}"
                os.makedirs(product_subdir, exist_ok=True)
                # symlink (not copy - avoids duplicating large FITS
                # data) each selected file into product_subdir under
                # its basename, preferring the wisp-subtracted version
                # when available (see `wisp_subtracted_filename`), so
                # `asn_from_list` can be run against bare basenames
                # instead of full paths: an association's `expname`
                # containing path information triggers a `UserWarning`
                # from `jwst.associations` and complicates moving the
                # association around independently of `self.folder_name`
                basenames = []
                for file in product_filenames:
                    src = (
                        f"{self.folder_name}/{wisp_subtracted_filename(file)}"
                    )
                    basename = os.path.basename(src)
                    link = f"{self.folder_name}/{product_subdir}/{basename}"
                    # `lexists`, not `exists`: a broken symlink (stale
                    # target since removed) still counts as "exists" to
                    # `os.symlink`, which errors if anything - even a
                    # dangling link - already sits at that path
                    if not os.path.lexists(link):
                        os.symlink(src, link)
                    basenames.append(basename)
                file_list = " ".join(basenames)
                os.system(
                    f"cd {self.folder_name}/{product_subdir} && "
                    f"asn_from_list -o {filt}.json "
                    f"--product-name {product_name} {file_list}"
                )
        asn_dir = f"{self.folder_name}/asn_{input_crds}/"
        galfind_logger.info(
            f"Generated associations for {self.survey} {self.pid=} in "
            f"{asn_dir}"
        )
        return list(groups.keys())

    @run_in_self_dir(lambda self: self.folder_name)
    @log_time(logging.INFO, u.hour)
    def run_stage1(
        self: Self,
        input_crds: int = 1584,
        steps: Dict[str, Any] = {},
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = "stage1.asdf",
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
        silence_pipeline: bool = True,
    ) -> None:
        """Run JWST stage 1 detector calibration pipeline.

        Applies detector-level corrections (bias, dark, nonlinearity) to
        uncalibrated raw data, producing rate images.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version. Default is 1584.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is "stage1.asdf".
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        silence_pipeline : `bool`, optional
            Whether to suppress the JWST pipeline's own console output
            for each file, showing a `tqdm` bar of files completed
            instead. Default is True.
        """
        if not hasattr(self, "n_files") or not hasattr(self, "n_uncals"):
            raise MissingDataError(
                f"{self.pid=} has no `n_files`/`n_uncals`; `download` "
                "and `move_uncals` must be run before `run_stage1`."
            )
        if self.n_files != self.n_uncals:
            warnings.warn(
                f"Expected {self.n_files} UNCAL files but only "
                f"{self.n_uncals} are fully downloaded in 'uncals/' "
                f"for {self.pid=}; proceeding with stage 1 regardless."
            )

        from jwst.pipeline import Detector1Pipeline

        self.run(
            Detector1Pipeline,
            search_str="uncals/*_uncal.fits",
            output_suffix="rate",
            input_crds=input_crds,
            steps=steps,
            config_file=config_file,
            asdf_savename=asdf_savename,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
            overwrite=overwrite,
            silence_pipeline=silence_pipeline,
        )

    @run_in_self_dir(lambda self: self.folder_name)
    @log_time(logging.INFO, u.hour)
    def run_stage2(
        self: Self,
        input_crds: int = 1584,
        steps: Dict[str, Any] = {},
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = "stage2.asdf",
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
        wisp_author_year: Optional[str] = "Sunnquist2024",
        silence_pipeline: bool = True,
    ) -> None:
        """Run JWST stage 2 image calibration pipeline.

        Applies image-level corrections (flat-field, photometry, WCS) with
        optional
        wavefront sensing and phase retrieval (WISP) processing.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version. Default is 1584.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is "stage2.asdf".
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        wisp_author_year : `str` or `None`, optional
            WISP processing to apply, if any. Default is "Sunnquist2024".
        silence_pipeline : `bool`, optional
            Whether to suppress the JWST pipeline's own console output
            for each file, showing a `tqdm` bar of files completed
            instead. Default is True.
        """
        from jwst.pipeline import Image2Pipeline

        # if wisp_when == "pre":
        #     from dewispify.dewisp_stage2 import (
        #         Image2PipelinePreDewisp as Image2PipelineDewisp,
        #     )
        # elif wisp_when == "post":
        #     from dewispify.dewisp_stage2 import (
        #         Image2PipelinePostDewisp as Image2PipelineDewisp,
        #     )
        # else:
        #     raise InvalidOptionError(
        #         f"wisp_when={wisp_when!r} not in ['pre', 'post']."
        #     )
        # ensure steps has a "wisps" entry
        # if "wisps" not in steps.keys():
        #     steps["wisps"] = {}
        # if "wisps" in steps:
        #     steps["wisps"]["wisp_when"] = wisp_when
        #     steps_wisp_when = steps["wisps"].get("wisp_when", None)
        #     if steps_wisp_when is not None:
        #         if steps_wisp_when != wisp_when:
        #             galfind_logger.warning(
        #                 f"Overriding {steps['wisps']['wisp_when']=} with "
        #                 f"{wisp_when=}"
        #             )

        self.run(
            Image2Pipeline,  # Dewisp,
            search_str=f"rate_{input_crds}/*_rate.fits",
            output_suffix="cal",
            input_crds=input_crds,
            steps=steps,
            config_file=config_file,
            asdf_savename=asdf_savename,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
            overwrite=overwrite,
            silence_pipeline=silence_pipeline,
        )

        if wisp_author_year is None:
            galfind_logger.info(
                "Skipping WISP processing for stage 2 output "
                "(wisp_author_year=None)"
            )
        elif wisp_author_year == "Sunnquist2024":
            from . import subtract_wisps

            galfind_logger.info(
                "Applying Sunnquist2024 WISP processing to stage 2 output"
            )
            post_stage2_files = glob.glob(f"cal_{input_crds}/*_cal.fits")
            subtract_wisps.subtract_Sunnquist24_wisps(
                post_stage2_files,
                n_cores=n_cores,
                plot=False,
                overwrite=overwrite,
            )
        else:
            raise InvalidOptionError(
                f"wisp_author_year={wisp_author_year!r} not in "
                "['Sunnquist2024', None]."
            )

    @run_in_self_dir(lambda self: self.folder_name)
    @log_time(logging.INFO, u.hour)
    def run_stage3(
        self: Self,
        input_crds: int = 1584,
        steps: Dict[str, Any] = {},
        asn_file_search_dir: Optional[str] = None,
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = "stage3.asdf",
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
        silence_pipeline: bool = True,
    ) -> Dict[str, Path]:
        """Run JWST stage 3 image processing pipeline.

        Performs image alignment, astrometric refinement, image stack creation,
        and source catalog extraction for multi-exposure datasets.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version. Default is 1584.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        asn_file_search_dir : `str`, optional
            Subdirectory under ``asn_{input_crds}/`` to search for ASN files.
            Default is None, which searches the top-level ASN directory.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is "stage3.asdf".
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        silence_pipeline : `bool`, optional
            Whether to suppress the JWST pipeline's own console output
            for each file, showing a `tqdm` bar of files completed
            instead. Default is True.

        Returns
        -------
        `dict` of `str` -> `Path`
            Each association file matched by `asn_file_search_dir`,
            mapped to its (now-existing) i2d output path - whether
            produced by this call or already present from an earlier
            one. Also accumulated, across every `run_stage3` call made
            on this instance, in `self.stage3_output_paths[output_dir]`.
        """
        from jwst.pipeline import Image3Pipeline

        if asn_file_search_dir is None:
            search_str = f"asn_{input_crds}/*/*.json"
        else:
            search_str = f"asn_{input_crds}/{asn_file_search_dir}/*.json"

        return self.run(
            Image3Pipeline,
            search_str=search_str,
            output_suffix="science",
            input_crds=input_crds,
            steps=steps,
            config_file=config_file,
            asdf_savename=asdf_savename,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
            overwrite=overwrite,
            silence_pipeline=silence_pipeline,
        )

    @run_in_self_dir(lambda self: self.folder_name)
    @log_time(logging.INFO, u.hour)
    def run(
        self: Self,
        pipe_cls: Type[JWSTPipeline],
        search_str: str,
        output_suffix: str,
        input_crds: int = 1584,
        steps: Dict[str, Any] = {},
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = None,
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
        silence_pipeline: bool = True,
    ) -> Union[List[Optional[str]], Dict[str, Path]]:
        """Execute a JWST pipeline stage on input data files.

        Generic pipeline runner that applies the specified JWST pipeline to
        matching input files with optional parallel processing.

        Parameters
        ----------
        pipe_cls : `Type[JWSTPipeline]`
            JWST pipeline class (Detector1Pipeline, Image2Pipeline, etc.).
        search_str : `str`
            Glob pattern to match input files.
        output_suffix : `str`
            Suffix for output files (e.g., "rate", "cal", "science").
        input_crds : `int`, optional
            CRDS context version. Default is 1584.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is None.
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        silence_pipeline : `bool`, optional
            Whether to suppress the JWST pipeline's own console output
            for each file, showing a `tqdm` bar of files completed
            instead. Default is True.

        Returns
        -------
        `list` of `str` or `None`, or `dict` of `str` -> `Path`
            For `pipe_cls=Image3Pipeline`: each matched association
            file mapped to its i2d output path (see `run_stage3`). For
            every other stage: one entry per freshly-processed file
            (already-complete files skipped by the pre-filter below
            aren't included), `None` where that file's `_call_stage`
            call didn't return an output path.
        """
        self.set_crds_context(input_crds)
        os.environ["CRDS_PATH"] = (
            f"{config['DEFAULT']['GALFIND_DATA']}/crds_cache"
        )

        # retrieve all files in this directory
        filenames = glob.glob(search_str)
        if len(filenames) == 0:
            galfind_logger.critical(
                f"No files found in {os.getcwd()}/{search_str}!"
            )
            return
        else:
            galfind_logger.info(
                f"Found {len(filenames)} {repr(self)} "
                + search_str.split("*")[-1]
                .replace("_", "")
                .replace(".fits", "")
                + " files for processing!"
            )

        if pre_download_refs:
            self.pre_download_refs(filenames, input_crds=input_crds)

        output_dir = f"{output_suffix}_{input_crds}"
        os.makedirs(output_dir, exist_ok=True)
        # write asdf
        if (
            asdf_savename is not None
            and not Path(asdf_savename).is_file()
            or overwrite
        ):
            # make stage 1 pipeline object
            if config_file is not None:
                galfind_logger.info(f"Loading config file: {config_file}")
                if not Path(config_file).is_file():
                    raise MissingFileError(
                        f"config_file={config_file!r} does not exist."
                    )
                if steps != {}:
                    galfind_logger.warning(
                        f"{steps=} ignored when using a config file."
                    )
                pipe = pipe_cls.from_config_file(config_file)
            else:
                pipe = pipe_cls(steps=steps)
            pipe.output_dir = output_dir
            pipe.export_config(asdf_savename)
            galfind_logger.info(
                f"Saved {pipe.__class__.__name__} pipeline configuration for "
                + f"{self.instrument.__class__.__name__} {self.pid=} "
                + f"to {os.getcwd()}/{asdf_savename}!"
            )

        # skip files whose output already exists, using a bare
        # existence check (not `is_complete_pipeline_output`, which
        # opens every file to verify its ASDF metadata) so this stays
        # cheap even for thousands of files; a file left truncated by
        # an earlier interrupted run won't be caught by this and needs
        # to be removed manually to be reprocessed. Any real failure
        # during processing is instead caught per-file below and
        # reported once every file has been attempted.
        already_done = 0
        to_process = []
        if pipe_cls.__name__ == "Image3Pipeline":
            # stage 3's output filename comes from the association's
            # declared product name, not a suffix-replace on the
            # input (association) filename like `stage_output_path`
            # assumes - see `stage3_output_paths`
            output_paths = stage3_output_paths(
                filenames, output_dir, cache=self.stage3_output_paths
            )
            for filename in filenames:
                if not overwrite and output_paths[filename].is_file():
                    already_done += 1
                else:
                    to_process.append(filename)
        else:
            for filename in filenames:
                out_path = stage_output_path(
                    filename, output_dir, output_suffix
                )
                if out_path.is_file():
                    already_done += 1
                else:
                    to_process.append(filename)
        if already_done:
            galfind_logger.info(
                f"{already_done}/{len(filenames)} {output_suffix!r} "
                f"files already exist for {self.pid=}; running "
                f"{pipe_cls.__name__} on the remaining "
                f"{len(to_process)}"
            )

        failed_files: List[str] = []
        if n_cores > 1:
            from multiprocessing import Pool

            tasks = [
                (
                    pipe_cls,
                    filename,
                    steps,
                    output_dir,
                    output_suffix,
                    silence_pipeline,
                )
                for filename in to_process
            ]

            with Pool(n_cores) as pool:
                outputs = []
                # `imap_unordered` (not `starmap`, which blocks and
                # only returns once every task is done) yields each
                # result as soon as it completes, so the tqdm bar
                # reflects real per-file progress rather than jumping
                # straight to 100% at the very end. `_unordered`
                # specifically (over plain `imap`) matters here: since
                # nothing downstream depends on output order, a slow
                # file shouldn't block the bar from advancing past
                # others that finish before it.
                for file, output, err in tqdm(
                    pool.imap_unordered(self._call_stage_star, tasks),
                    desc=f"Running {pipe_cls.__name__} with {self.pid=}, "
                    f"CRDS={os.environ['CRDS_CONTEXT']}, and {n_cores=}",
                    initial=already_done,
                    total=len(filenames),
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                ):
                    outputs.append(output)
                    # note failures but keep running the rest of the
                    # files; only raise once everything has been tried
                    if err is not None:
                        galfind_logger.error(err)
                        failed_files.append(file)
        else:
            outputs = np.full(len(to_process), None, dtype=object)
            for i, filename in tqdm(
                enumerate(to_process),
                desc=(
                    f"Running {pipe_cls.__name__} with {self.pid=} and "
                    f"CRDS={os.environ['CRDS_CONTEXT']}"
                ),
                initial=already_done,
                total=len(filenames),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            ):
                _, output, err = self._call_stage(
                    pipe_cls,
                    filename,
                    steps,
                    output_dir,
                    output_suffix,
                    silence_pipeline,
                )
                outputs[i] = output
                # note failures but keep running the rest of the
                # files; only raise once everything has been tried
                if err is not None:
                    galfind_logger.error(err)
                    failed_files.append(filename)

        if failed_files:
            raise GalfindError(
                f"{len(failed_files)}/{len(to_process)} file(s) failed "
                f"during {pipe_cls.__name__} for {self.pid=}: "
                f"{failed_files}"
            )
        if pipe_cls.__name__ == "Image3Pipeline":
            return output_paths
        return outputs

    @run_in_self_dir(lambda self: self.folder_name)
    def ingest_stage3_outputs(
        self: Self,
        version: str,
        instrument: Type[Instrument] = NIRCam,
    ) -> Dict[str, List[Path]]:
        """Symlink this instance's stage 3 i2d outputs for use in a `Data`.

        Bridges `run_stage3`'s PID-based output layout
        (``{GALFIND_DATA}/jwst/PID={pid}/science_{input_crds}/*_i2d.fits``,
        tracked in `self.stage3_output_paths`) into the survey/version/
        pixel-scale layout `Data.pipeline`/`Data.from_survey_version_psfs`
        expect (``{GALFIND_DATA}/{facility}/{survey}/{Instrument}/
        {version}/{pixscale}/``).

        For each i2d file, the survey (pointing) name is parsed from
        its filename (`parse_i2d_filename`) and its pixel scale is read
        from its own WCS (`i2d_pixel_scale`) - both determined per-file,
        since a single instance's stage 3 outputs can cover several
        pointings and (in principle) different pixel scales. `version`
        is not derived from the files themselves (nothing in an i2d
        file's name or header records the stage 1/2/3 step parameters
        that produced it) - pass the alias from
        `get_or_register_reduction_version`.

        A destination already symlinked to the same real source file is
        left alone; one that exists but points elsewhere (e.g. a stale
        link from a previous, now-superseded reprocessing) is replaced.

        Returns
        -------
        `dict` of `str` -> `list` of `Path`
            Every survey (pointing) name touched, mapped to the
            symlinks created/kept for it - i.e. what
            ``Data.pipeline(survey, version)`` can now be called with,
            for each survey in the returned keys.
        """
        instrument_instance = instrument()
        symlinks_by_survey: Dict[str, List[Path]] = {}
        i2d_files = [
            path
            for output_paths in self.stage3_output_paths.values()
            for path in output_paths.values()
        ]
        for i2d_file in i2d_files:
            i2d_file = Path(i2d_file).resolve()
            survey, _ = parse_i2d_filename(i2d_file)
            pix_scale = i2d_pixel_scale(i2d_file)

            target_dir = Path(
                Data._get_data_dir(
                    survey, version, instrument_instance, pix_scale
                )
            )
            link = target_dir / i2d_file.name

            if link.is_symlink() and link.resolve() == i2d_file:
                galfind_logger.debug(f"{link} already links to {i2d_file}")
            else:
                if os.path.lexists(link):
                    galfind_logger.warning(
                        f"Replacing stale {link} (was pointing elsewhere) "
                        f"with a link to {i2d_file}"
                    )
                    link.unlink()
                os.symlink(i2d_file, link)
                galfind_logger.info(f"Symlinked {i2d_file} -> {link}")

            symlinks_by_survey.setdefault(survey, []).append(link)
        return symlinks_by_survey

    @staticmethod
    def _call_stage_star(
        args: Tuple,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """`Pool.imap`-compatible wrapper for `_call_stage`.

        `multiprocessing.Pool` has no starmap variant that yields
        results incrementally as they complete (`starmap` blocks and
        only returns once every task is done), so real per-file
        progress needs `imap`, which requires a single-argument
        callable - this unpacks the tuple and forwards it.
        """
        return Raw_JWST_Data._call_stage(*args)

    @staticmethod
    def _call_stage(
        pipe_cls: Type[JWSTPipeline],
        file: str,
        steps: Dict[str, Any],
        output_dir: str,
        output_suffix: str,
        silence_pipeline: bool = True,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Run pipeline on a single file."""
        # TODO: Generalize this to work for stage 3 as well!
        output = None
        err = None
        out_path = stage_output_path(file, output_dir, output_suffix)
        try:
            # classmethod
            call_kwargs = dict(
                steps=steps, output_dir=output_dir, save_results=True
            )
            if silence_pipeline:
                with silence_stdio():
                    model = pipe_cls.call(file, **call_kwargs)
            else:
                model = pipe_cls.call(file, **call_kwargs)
            # `save_results=True` already wrote the output to disk,
            # and nothing downstream needs the in-memory model, so
            # close it rather than returning it: an `ImageModel`
            # holds an open file handle internally, which can't be
            # pickled back to the parent process through a
            # `multiprocessing.Pool`. Some pipeline stages (e.g.
            # those driven by an association file) return a list
            # of models rather than a single one, and Image3Pipeline
            # (stage 3) returns `None` instead of a model at all, since
            # its outputs are mosaics/catalogs written directly to disk.
            models = model if isinstance(model, list) else [model]
            for m in models:
                if m is not None:
                    m.close()
            output = str(out_path)
        except Exception as e:
            err = (
                "\n--- ERROR PROCESSING FILE ---\n"
                + f"File: {file}\n"
                + f"Error Type: {type(e).__name__}\n"
                + f"Error Message: {e}\n"
                + f"Traceback:\n{traceback.format_exc()}"
            )
        return file, output, err

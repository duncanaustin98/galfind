"""Automatically maintained README files for galfind FITS catalogues.

Every time a galfind processing step writes to a catalogue FITS file,
`update_cat_readme` is called to regenerate a human-readable
``<catalogue>_README.txt`` next to it. The README documents:

- the catalogue provenance (survey, version, instrument, bands, apertures),
- a processing history: one entry per step, recording which HDUs/columns
  were created, added or removed and when,
- every HDU extension, its header metadata and a description of each
  column (with per-band columns grouped together).

The processing history and the last-seen column layout are stored in a
small ``<catalogue>_README.json`` sidecar, which is used to work out what
changed between successive writes.
"""

from __future__ import annotations

import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astropy.io import fits

from .. import config, galfind_logger

README_SUFFIX = "_README.txt"
STATE_SUFFIX = "_README.json"
_LINE_WIDTH = 100
_MAJOR_DIVIDER = "=" * _LINE_WIDTH
_MINOR_DIVIDER = "-" * _LINE_WIDTH
_MAX_HISTORY_TEMPLATES = 12

# standard FITS table header keywords that carry no provenance information
_STANDARD_HDR_KEYS = re.compile(
    r"^(XTENSION|BITPIX|NAXIS\d*|PCOUNT|GCOUNT|TFIELDS|EXTNAME|EXTVER|"
    r"T(TYPE|FORM|UNIT|DIM|NULL|DISP|SCAL|ZERO)\d+|COMMENT|HISTORY|)$"
)
_FITS_DTYPES = {
    "L": "bool",
    "B": "uint8",
    "I": "int16",
    "J": "int32",
    "K": "int64",
    "E": "float32",
    "D": "float64",
    "C": "complex64",
    "M": "complex128",
    "A": "str",
}
_PERCENTILES = ("16", "50", "84")

# (HDU name regex, column template regex, description). The first match
# wins, so more specific patterns must come before more general ones.
# Descriptions are formatted with the regex groups ({0}, {1}, ...) and
# the OBJECTS header aperture diameters ({aper}) and, for ``{pc}`` column
# groups, the posterior percentiles present ({pcs}).
_COLUMN_DESCRIPTIONS: List[Tuple[str, str, str]] = [
    # ---------------------------------------------------------------- IDs
    (
        r"^OBJECTS$",
        r"^NUMBER$",
        "Unique source ID from the forced photometry detection image",
    ),
    (
        r".*",
        r"^(NUMBER|ID|IDENT|#ID)$",
        "Source ID (matches NUMBER in OBJECTS)",
    ),
    # -------------------------------------------------- OBJECTS: photometry
    (r"^OBJECTS$", r"^[XY]_IMAGE$", "Pixel position in the detection image"),
    (r"^OBJECTS$", r"^ALPHA_J2000$", "Right ascension (J2000) [deg]"),
    (r"^OBJECTS$", r"^DELTA_J2000$", "Declination (J2000) [deg]"),
    (
        r"^OBJECTS$",
        r"^FLUX_APER_\{band\}$",
        "Aperture flux in {aper} diameter apertures [image units; "
        "convert using the band zero point]",
    ),
    (
        r"^OBJECTS$",
        r"^FLUXERR_APER_\{band\}$",
        "Aperture flux error from the forced photometry error map "
        "[image units]",
    ),
    (
        r"^OBJECTS$",
        r"^MAG_APER_\{band\}$",
        "Aperture magnitude in {aper} diameter apertures [AB mag]; "
        "99 where flux <= 0",
    ),
    (
        r"^OBJECTS$",
        r"^MAGERR_APER_\{band\}$",
        "Aperture magnitude error [AB mag]",
    ),
    # --------------------------------------- OBJECTS: aperture corrections
    (
        r"^OBJECTS$",
        r"^MAG_APER_\{band\}_aper_corr$",
        "Aperture magnitude with the point-source PSF aperture correction "
        "applied [AB mag]",
    ),
    (
        r"^OBJECTS$",
        r"^FLUX_APER_\{band\}_aper_corr$",
        "Aperture flux with the point-source PSF aperture correction "
        "applied [image units]",
    ),
    (
        r"^OBJECTS$",
        r"^FLUX_APER_\{band\}_aper_corr_Jy$",
        "Aperture flux with the point-source PSF aperture correction "
        "applied [Jy]",
    ),
    # ----------------------------------------------------- OBJECTS: masking
    (
        r"^OBJECTS$",
        r"^unmasked_\{band\}$",
        "True if the source centre is unmasked (and within the image) in "
        "this band",
    ),
    # ------------------------------------------------ OBJECTS: local depths
    (
        r"^OBJECTS$",
        r"^loc_depth_\{band\}$",
        "Local 5 sigma depth from nearby empty apertures [AB mag]",
    ),
    (
        r"^OBJECTS$",
        r"^d_(\d+)_\{band\}$",
        "Distance to the {0}th nearest empty aperture used to calculate the "
        "local depth [pix]",
    ),
    (
        r"^OBJECTS$",
        r"^n_aper_(.+)_\{band\}$",
        "Number of empty apertures within {0} pix used to calculate the "
        "local depth",
    ),
    (
        r"^OBJECTS$",
        r"^sigma_\{band\}$",
        "Detection significance (S/N) of the aperture flux given the local "
        "depth",
    ),
    (
        r"^OBJECTS$",
        r"^FLUXERR_APER_\{band\}_loc_depth$",
        "Aperture flux error from the local depth (5 sigma depth flux / 5) "
        "[image units]",
    ),
    (
        r"^OBJECTS$",
        r"^FLUXERR_APER_\{band\}_loc_depth_(\d+)pc_Jy$",
        "Local depth aperture flux error, with a {0}% minimum fractional "
        "error floor imposed [Jy]",
    ),
    # ------------------------------------ OBJECTS: extended source corrections
    (
        r".*",
        r"^ext_src_corr_(.+)_\{band\}$",
        "Extended source correction (total / aperture flux) for {0} "
        "diameter apertures",
    ),
    # ------------------------------------------- OBJECTS: other SExtractor
    (
        r"^OBJECTS$",
        r"^(MAG|MAGERR|FLUX|FLUXERR)_(AUTO|BEST|ISO|ISOCOR|PETRO|WIN)"
        r"_\{band\}$",
        "SExtractor {0}_{1} (see the SExtractor documentation)",
    ),
    (
        r"^OBJECTS$",
        r"^([A-Z][A-Z0-9_]*)_\{band\}$",
        "SExtractor {0} parameter (see the SExtractor documentation)",
    ),
    # --------------------------------------------------------------- EAZY
    (r"^EAZY", r"^zbest_16(_.*)?$", "16th percentile of the redshift PDF"),
    (r"^EAZY", r"^zbest_84(_.*)?$", "84th percentile of the redshift PDF"),
    (r"^EAZY", r"^zbest(_.*)?$", "Best-fitting photometric redshift"),
    (r"^EAZY", r"^chi2_best(_.*)?$", "Chi-squared of the best-fitting SED"),
    (
        r"^EAZY",
        r"^([A-Za-z]+)_rf_flux_err(_.*)?$",
        "Error on the rest-frame {0}-band flux",
    ),
    (
        r"^EAZY",
        r"^([A-Za-z]+)_rf_flux(_.*)?$",
        "Rest-frame {0}-band flux from the best-fitting SED",
    ),
    # ------------------------------------------------------------ LePhare
    (r"^LEPHARE", r"^Z_BEST(_.*)?$", "Best-fitting photometric redshift"),
    (r"^LEPHARE", r"^CHI_BEST(_.*)?$", "Chi-squared of the best-fitting SED"),
    # -------------------------------------------------------- Brown dwarfs
    (r"^BROWN_DWARF", r"^best_template$", "Best-fitting brown dwarf template"),
    (r"^BROWN_DWARF", r"^best_library$", "Template library of best fit"),
    (r"^BROWN_DWARF", r"^chi2$", "Chi-squared of the best-fitting template"),
    (r"^BROWN_DWARF", r"^red_chi2$", "Reduced chi-squared of the best fit"),
    (r"^BROWN_DWARF", r"^template_norm$", "Normalisation of the best fit"),
    (r"^BROWN_DWARF", r"^temp$", "Best-fitting effective temperature [K]"),
    (r"^BROWN_DWARF", r"^log_g$", "Best-fitting surface gravity, log(g)"),
    (
        r"^BROWN_DWARF",
        r"^\{band\}_nJy$",
        "Flux of the best-fitting template in this band [nJy]",
    ),
    (
        r"^BROWN_DWARF",
        r"^(.+)$",
        "Best-fitting template parameter '{0}'",
    ),
    # ----------------------------------------------------------- Bagpipes
    (
        r"^BAGPIPES",
        r"^input_redshift$",
        "Input redshift (prior centre) used in the fit",
    ),
    (r"^BAGPIPES", r"^log_evidence$", "Bayesian log-evidence of the fit"),
    (r"^BAGPIPES", r"^log_evidence_err$", "Error on the log-evidence"),
    (
        r"^BAGPIPES",
        r"^chisq_phot$",
        "Chi-squared of the maximum-likelihood fit to the photometry",
    ),
    (r"^BAGPIPES", r"^n_bands$", "Number of photometric bands fitted"),
    # ------------------------------------------- generic percentile groups
    (
        r".*",
        r"^(.+)_\{pc\}$",
        "{pcs} percentiles of the '{0}' posterior",
    ),
    # ------------------------------------------------- derived properties
    (r"^PROPERTIES", r"^(.+)_l1$", "Lower 1 sigma error on {0}"),
    (r"^PROPERTIES", r"^(.+)_u1$", "Upper 1 sigma error on {0}"),
    (
        r"^PROPERTIES",
        r"^(.+)_n_UV_bands$",
        "Number of rest-frame UV bands used to calculate {0}",
    ),
    (
        r"^PROPERTIES",
        r"^(.+)_negative_flux_pc$",
        "Percentage of rest-frame UV bands with negative flux used for {0}",
    ),
    (
        r"^PROPERTIES",
        r"^(.+)_rest_UV_filt_names$",
        "Rest-frame UV bands used to calculate {0}",
    ),
    (
        r"^PROPERTIES",
        r"^beta_\[(\d+),(\d+)\]AA.*$",
        "UV continuum slope, beta, fitted over rest-frame {0}-{1} Angstrom",
    ),
    (
        r"^PROPERTIES",
        r"^m1500_.*$",
        "Apparent magnitude at rest-frame 1500 Angstrom [AB mag]",
    ),
    (
        r"^PROPERTIES",
        r"^M1500_.*$",
        "Absolute magnitude at rest-frame 1500 Angstrom (M_UV) [AB mag]",
    ),
    # ------------------------------------------------------------ selection
    (
        r"^SELECTION$",
        r"^(.+)__(.+)$",
        "Diagnostic '{1}' for the '{0}' selection",
    ),
]

# (HDU name regex, description). First match wins.
_HDU_DESCRIPTIONS: List[Tuple[str, str]] = [
    (
        r"^OBJECTS$",
        "Forced photometry catalogue (one row per detected source), plus "
        "columns appended by subsequent galfind steps (aperture "
        "corrections, masking, local depths, ...).",
    ),
    (r"^EAZY", "EAZY-py SED fitting results."),
    (r"^LEPHARE", "LePhare SED fitting results."),
    (r"^BAGPIPES", "Bagpipes SED fitting posterior summaries."),
    (r"^BROWN_DWARF", "Brown dwarf template fitting results."),
    (
        r"^PROPERTIES_(.+)$",
        "Derived galaxy properties calculated using the redshifts from the "
        "{0} SED fitting HDU.",
    ),
    (
        r"^SELECTION$",
        "Boolean flags (and their diagnostics) recording whether each "
        "source passes the galfind Selector criteria named in the columns.",
    ),
]


def get_readme_path(cat_path: str) -> str:
    """Path to the README text file for the catalogue at `cat_path`."""
    return f"{_strip_fits_ext(cat_path)}{README_SUFFIX}"


def _get_state_path(cat_path: str) -> str:
    return f"{_strip_fits_ext(cat_path)}{STATE_SUFFIX}"


def _strip_fits_ext(cat_path: str) -> str:
    return re.sub(r"\.fits(\.gz)?$", "", str(cat_path))


def update_cat_readme(
    cat_path: str,
    step: Optional[str] = None,
    details: Optional[str] = None,
) -> None:
    """Regenerate the README for a FITS catalogue after it has been written.

    Compares the current HDU/column layout of `cat_path` against the layout
    recorded the last time the README was written, appends an entry to the
    processing history describing what changed, then re-writes
    ``<catalogue>_README.txt``. Any failure is logged and swallowed so that
    README generation can never interrupt the main galfind pipeline.

    Parameters
    ----------
    cat_path : `str`
        Path to the FITS catalogue that has just been written.
    step : `str`, optional
        Name of the processing step that wrote to the catalogue. If `None`,
        the step is inferred from the HDUs and columns that changed, and no
        history entry is made if nothing changed. Default is `None`.
    details : `str`, optional
        Extra free-text information to record in the history entry.
        Default is `None`.
    """
    if not config.getboolean("Other", "MAKE_CAT_README", fallback=True):
        return
    try:
        _update_cat_readme(str(cat_path), step, details)
    except Exception as e:
        galfind_logger.warning(
            f"Failed to update catalogue README for {cat_path}: {e!r}"
        )


def _update_cat_readme(
    cat_path: str,
    step: Optional[str],
    details: Optional[str],
) -> None:
    if not Path(cat_path).is_file():
        return
    structure = _read_fits_structure(cat_path)
    state_path = _get_state_path(cat_path)
    if Path(state_path).is_file():
        with open(state_path, "r") as f:
            state = json.load(f)
    else:
        state = {"history": [], "hdus": None}
    bands = _get_band_names(structure)

    changes = _diff_structure(state["hdus"], structure, bands)
    if step is None and state["hdus"] is None:
        # the steps that made this catalogue pre-date the README, so only
        # record which HDUs already existed (columns are documented below)
        step = "README initialised (catalogue already existed)"
        for change in changes:
            change["added"] = []
    elif step is None and len(changes) > 0:
        step = _infer_step(changes)
    if step is not None:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "galfind_version": _get_galfind_version(),
            "changes": changes,
        }
        if details is not None:
            entry["details"] = details
        state["history"].append(entry)
    state["hdus"] = {
        hdu_name: {
            "columns": [col["name"] for col in hdu_info["columns"]],
            "n_rows": hdu_info["n_rows"],
        }
        for hdu_name, hdu_info in structure.items()
    }

    readme_path = get_readme_path(cat_path)
    _atomic_write(state_path, json.dumps(state, indent=1))
    _atomic_write(
        readme_path, _render_readme(cat_path, structure, state, bands)
    )
    galfind_logger.debug(f"Updated catalogue README: {readme_path}")


def _atomic_write(path: str, text: str) -> None:
    from . import useful_funcs_austind as funcs

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        f.write(text)
    os.replace(tmp_path, path)
    funcs.change_file_permissions(path)


def _get_galfind_version() -> str:
    try:
        from .._version import __version__

        return __version__
    except ImportError:
        return "unknown"


# ------------------------------------------------------------------ parsing


def _read_fits_structure(cat_path: str) -> Dict[str, Dict[str, Any]]:
    """Read HDU names, row counts, columns and metadata from the headers
    of `cat_path`, without loading any table data."""
    structure = {}
    with fits.open(cat_path, memmap=True, lazy_load_hdus=True) as hdul:
        for hdu in hdul:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            hdr = hdu.header
            structure[hdu.name] = {
                "n_rows": int(hdr.get("NAXIS2", 0)),
                "columns": [
                    {
                        "name": col.name,
                        "dtype": _fits_format_to_dtype(col.format),
                        "unit": col.unit,
                    }
                    for col in hdu.columns
                ],
                "meta": {
                    key: hdr[key]
                    for key in dict.fromkeys(hdr.keys())
                    if not _STANDARD_HDR_KEYS.match(key)
                },
            }
    return structure


def _fits_format_to_dtype(fits_format: str) -> str:
    match = re.match(r"^(\d*)([A-Z])", str(fits_format))
    if match is None:
        return str(fits_format)
    repeat, code = match.groups()
    dtype = _FITS_DTYPES.get(code, code)
    if code == "A":
        return f"str{repeat}" if repeat else "str"
    if repeat not in ("", "1"):
        return f"{dtype}[{repeat}]"
    return dtype


def _get_band_names(structure: Dict[str, Dict[str, Any]]) -> List[str]:
    """Band names in the catalogue, from the OBJECTS 'BANDS' header and
    the raw per-band aperture photometry columns (which also include any
    stacked forced photometry band)."""
    bands = []
    if "OBJECTS" in structure:
        objects = structure["OBJECTS"]
        try:
            bands.extend(ast.literal_eval(str(objects["meta"]["BANDS"])))
        except (KeyError, ValueError, SyntaxError):
            pass
        colnames = {col["name"] for col in objects["columns"]}
        for name in colnames:
            if name.startswith("FLUX_APER_"):
                band = name[len("FLUX_APER_") :]
                if (
                    f"FLUXERR_APER_{band}" in colnames
                    and f"MAGERR_APER_{band}" in colnames
                ):
                    bands.append(band)
    return _order_bands(set(bands), objects["meta"]) if bands else []


def _templatise(
    colname: str, band_regex: Optional[re.Pattern]
) -> Tuple[str, Optional[str], Optional[str]]:
    """Collapse a column name into a template, replacing its band name with
    ``{band}`` or its posterior percentile suffix with ``{pc}``.

    Returns
    -------
    template, band, percentile
    """
    if band_regex is not None:
        match = band_regex.search(colname)
        if match is not None:
            template = (
                colname[: match.start()] + "{band}" + colname[match.end() :]
            )
            return template, match.group(0), None
    pc_match = re.match(rf"^(.+)_({'|'.join(_PERCENTILES)})$", colname)
    if pc_match is not None:
        return f"{pc_match.group(1)}_{{pc}}", None, pc_match.group(2)
    return colname, None, None


def _make_band_regex(bands: List[str]) -> Optional[re.Pattern]:
    if len(bands) == 0:
        return None
    # longest first so stacked/underscored band names match before their
    # constituent band names
    alternation = "|".join(
        re.escape(band) for band in sorted(bands, key=len, reverse=True)
    )
    # bands are only matched as whole '_'-separated tokens
    return re.compile(rf"(?:(?<=_)|^)(?:{alternation})(?=_|$)")


def _group_columns(
    columns: List[Dict[str, Any]],
    bands: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Group columns by template, preserving first-appearance order."""
    band_regex = _make_band_regex(bands)
    groups = {}
    for col in columns:
        template, band, pc = _templatise(col["name"], band_regex)
        group = groups.setdefault(
            template,
            {
                "names": [],
                "bands": [],
                "percentiles": [],
                "dtype": col["dtype"],
                "unit": col["unit"],
            },
        )
        group["names"].append(col["name"])
        if band is not None:
            group["bands"].append(band)
        if pc is not None:
            group["percentiles"].append(pc)
    # a lone percentile column is not a group; restore its real name
    for template in list(groups.keys()):
        group = groups[template]
        if "{pc}" in template and len(group["names"]) == 1:
            groups[group["names"][0]] = groups.pop(template)
            groups[group["names"][0]]["percentiles"] = []
    return groups


def _diff_structure(
    prev_hdus: Optional[Dict[str, Dict[str, Any]]],
    structure: Dict[str, Dict[str, Any]],
    bands: List[str],
) -> List[Dict[str, Any]]:
    """Describe the HDU/column changes between two catalogue layouts."""
    prev_hdus = {} if prev_hdus is None else prev_hdus
    changes = []
    for hdu_name, hdu_info in structure.items():
        colnames = [col["name"] for col in hdu_info["columns"]]
        if hdu_name not in prev_hdus:
            changes.append(
                {
                    "hdu": hdu_name,
                    "action": "created",
                    "added": _summarise_names(colnames, bands),
                    "removed": [],
                    "n_rows": [None, hdu_info["n_rows"]],
                }
            )
            continue
        prev = prev_hdus[hdu_name]
        prev_colnames = set(prev["columns"])
        added = [name for name in colnames if name not in prev_colnames]
        removed = [name for name in prev["columns"] if name not in colnames]
        if (
            len(added) > 0
            or len(removed) > 0
            or (prev["n_rows"] != hdu_info["n_rows"])
        ):
            changes.append(
                {
                    "hdu": hdu_name,
                    "action": "updated",
                    "added": _summarise_names(added, bands),
                    "removed": _summarise_names(removed, bands),
                    "n_rows": [prev["n_rows"], hdu_info["n_rows"]],
                }
            )
    for hdu_name, prev in prev_hdus.items():
        if hdu_name not in structure:
            changes.append(
                {
                    "hdu": hdu_name,
                    "action": "removed",
                    "added": [],
                    "removed": [],
                    "n_rows": [prev["n_rows"], None],
                }
            )
    return changes


def _summarise_names(names: List[str], bands: List[str]) -> List[str]:
    groups = _group_columns(
        [{"name": name, "dtype": "", "unit": None} for name in names], bands
    )
    return [
        f"{template} ({len(group['bands'])} bands)"
        if len(group["bands"]) > 1
        else template
        for template, group in groups.items()
    ]


def _infer_step(changes: List[Dict[str, Any]]) -> str:
    """Infer a human-readable processing step from catalogue changes."""
    steps = []
    for change in changes:
        hdu_name = change["hdu"]
        added = " ".join(change["added"])
        if change["action"] == "removed":
            step = f"Removed {hdu_name} HDU"
        elif hdu_name == "OBJECTS":
            if change["action"] == "created":
                step = "Forced photometry"
            elif "unmasked_" in added:
                step = "Masking"
            elif re.search(r"loc_depth|sigma_|(^| )d_\d+_|n_aper_", added):
                step = "Local depths"
            elif "aper_corr" in added:
                step = "Aperture corrections"
            elif "ext_src_corr" in added:
                step = "Extended source corrections"
            else:
                step = "Updated OBJECTS"
        elif hdu_name == "SELECTION":
            step = "Selection"
        elif hdu_name.startswith("PROPERTIES"):
            step = f"Property calculation ({hdu_name})"
        elif re.match(r"^(EAZY|LEPHARE|BAGPIPES|BROWN_DWARF)", hdu_name):
            step = f"SED fitting ({hdu_name})"
        else:
            step = f"Updated {hdu_name}"
        if step not in steps:
            steps.append(step)
    return "; ".join(steps)


# -------------------------------------------------------------- descriptions


def _describe_hdu(hdu_name: str) -> str:
    for pattern, description in _HDU_DESCRIPTIONS:
        match = re.match(pattern, hdu_name)
        if match is not None:
            return description.format(*match.groups())
    return "-"


def _describe_column(
    hdu_name: str,
    template: str,
    group: Dict[str, Any],
    aper: str,
) -> str:
    for hdu_pattern, col_pattern, description in _COLUMN_DESCRIPTIONS:
        if re.match(hdu_pattern, hdu_name) is None:
            continue
        match = re.match(col_pattern, template)
        if match is not None:
            pcs = "/".join(f"{pc}th" for pc in group["percentiles"])
            return description.format(*match.groups(), aper=aper, pcs=pcs)
    if hdu_name == "SELECTION" and group["dtype"] == "bool":
        return f"True if the source passes the '{template}' selection"
    return "-"


# ---------------------------------------------------------------- rendering


def _render_readme(
    cat_path: str,
    structure: Dict[str, Dict[str, Any]],
    state: Dict[str, Any],
    bands: List[str],
) -> str:
    objects_meta = structure.get("OBJECTS", {}).get("meta", {})
    aper = str(objects_meta.get("APERDIAM", "the catalogue"))
    lines = [
        _MAJOR_DIVIDER,
        f"GALFIND CATALOGUE README: {Path(cat_path).name}",
        _MAJOR_DIVIDER,
        "This file is generated automatically by galfind and is re-written "
        "every time a",
        "processing step writes to the catalogue - any manual edits will be "
        "overwritten.",
        "",
        f"Catalogue path:  {cat_path}",
        f"Last updated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"galfind version: {_get_galfind_version()}",
    ]
    for label, key in [
        ("Survey", "SURVEY"),
        ("Version", "VERSION"),
        ("Instrument(s)", "INSTR"),
        ("Aperture diams", "APERDIAM"),
        ("Phot. method", "METHODS"),
        ("Error type", "ERR_TYPE"),
    ]:
        if key in objects_meta:
            lines.append(f"{label + ':':<17}{objects_meta[key]}")
    if len(bands) > 0:
        lines.append(f"{'Bands:':<17}{', '.join(bands)}")
    if "OBJECTS" in structure:
        lines.append(f"{'N sources:':<17}{structure['OBJECTS']['n_rows']}")

    # processing history
    lines += ["", _MINOR_DIVIDER, "PROCESSING HISTORY", _MINOR_DIVIDER]
    if len(state["history"]) == 0:
        lines.append("(no processing steps recorded)")
    for i, entry in enumerate(state["history"]):
        lines.append(f"{i + 1:>3}. [{entry['time']}] {entry['step']}")
        if "details" in entry:
            lines.append(f"       {entry['details']}")
        for change in entry["changes"]:
            lines.extend(_render_change(change))

    # HDU summary
    lines += ["", _MINOR_DIVIDER, "CONTENTS", _MINOR_DIVIDER]
    name_width = max([len(name) for name in structure] + [8])
    lines.append(f"{'HDU':<{name_width}}  {'N_rows':>8}  {'N_cols':>6}")
    for hdu_name, hdu_info in structure.items():
        lines.append(
            f"{hdu_name:<{name_width}}  {hdu_info['n_rows']:>8}  "
            f"{len(hdu_info['columns']):>6}"
        )

    # per-HDU column descriptions
    for i, (hdu_name, hdu_info) in enumerate(structure.items()):
        lines += [
            "",
            _MINOR_DIVIDER,
            f"HDU {i + 1}: {hdu_name}  ({hdu_info['n_rows']} rows, "
            f"{len(hdu_info['columns'])} columns)",
            _MINOR_DIVIDER,
            f"Description: {_describe_hdu(hdu_name)}",
        ]
        if len(hdu_info["meta"]) > 0:
            lines.append("Header metadata:")
            for key, value in hdu_info["meta"].items():
                lines.append(f"    {key} = {value}")
        lines.append("Columns:")
        lines.extend(_render_columns(hdu_name, hdu_info, bands, aper))
    lines.append("")
    return "\n".join(lines)


def _order_bands(bands: set, objects_meta: Dict[str, Any]) -> List[str]:
    """Order bands as in the OBJECTS 'BANDS' header (blue -> red), with
    any bands not listed there (e.g. stacked detection bands) at the end."""
    try:
        header_bands = list(ast.literal_eval(str(objects_meta["BANDS"])))
    except (KeyError, ValueError, SyntaxError):
        header_bands = []
    return [band for band in header_bands if band in bands] + sorted(
        band for band in bands if band not in header_bands
    )


def _render_change(change: Dict[str, Any]) -> List[str]:
    n_prev, n_new = change["n_rows"]
    if change["action"] == "created":
        summary = f"created {change['hdu']} ({n_new} rows)"
    elif change["action"] == "removed":
        summary = f"removed {change['hdu']}"
    else:
        summary = f"updated {change['hdu']}"
        if n_prev != n_new:
            summary += f" (rows: {n_prev} -> {n_new})"
    lines = [f"       - {summary}"]
    for sign, templates in [("+", change["added"]), ("-", change["removed"])]:
        for template in templates[:_MAX_HISTORY_TEMPLATES]:
            lines.append(f"           {sign} {template}")
        if len(templates) > _MAX_HISTORY_TEMPLATES:
            lines.append(
                f"           {sign} ... and "
                f"{len(templates) - _MAX_HISTORY_TEMPLATES} more column groups"
            )
    return lines


def _render_columns(
    hdu_name: str,
    hdu_info: Dict[str, Any],
    bands: List[str],
    aper: str,
) -> List[str]:
    groups = _group_columns(hdu_info["columns"], bands)
    name_width = min(max([len(template) for template in groups] + [4]), 60)
    lines = []
    for template, group in groups.items():
        dtype = group["dtype"]
        if group["unit"]:
            dtype += f" [{group['unit']}]"
        description = _describe_column(hdu_name, template, group, aper)
        lines.append(
            f"    {template:<{name_width}}  {dtype:<14}  {description}"
        )
        if len(group["bands"]) > 0:
            group_bands = [band for band in bands if band in group["bands"]]
            lines.append(
                f"{' ' * (4 + name_width + 2)}bands: {', '.join(group_bands)}"
            )
    return lines

"""Tests for the `update` flag threaded through `Data.pipeline` into
`Data.perform_forced_phot` / `Data._combine_forced_phot_cats` and
`Data.append_loc_depth_cols` / `Depths.append_loc_depth_cols`.

`update=False` (the default) must still perform a stage's computation and
write the master catalogue the *first* time it's run for a survey/version
(when nothing exists on disk yet), but must skip re-computing and
re-writing it on subsequent runs once the catalogue already exists -- this
is what avoids both the redundant FITS rewrite and the concurrent-write
race in `Catalogue.update_fits_cat` (see `os.replace` there) on repeat
loads of an already-processed catalogue.
"""

import glob
import os
import shutil
from pathlib import Path

import pytest
from astropy.table import Table

# anchor to this file's own location, not the process cwd, so
# GALFIND_WORK/GALFIND_DATA always land under <repo_root>/testing/
# test_work regardless of the directory tests are invoked from
os.environ["GALFIND_CONFIG_DIR"] = os.path.dirname(os.path.abspath(__file__))
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

import galfind
from conftest import _mock_gaia_launch_job_async
from galfind.imaging import Data


@pytest.fixture
def update_test_version(version):
    # a version string distinct from the shared `data`/`cat` session
    # fixtures', so these tests build/rewrite their own catalogue rather
    # than the one those fixtures depend on. `version_to_dir_dict` below
    # maps it back to the real on-disk test data (Data._get_data_dir keys
    # off `version.split("_")[0]` when a mapping is supplied), while
    # `self.version` -- and therefore the output catalogue path -- keeps
    # the full, unique string.
    fake_version = f"{version}_update_test"
    yield fake_version
    work_dir = galfind.config["DEFAULT"]["GALFIND_WORK"]
    for path in glob.glob(f"{work_dir}/**/*{fake_version}*", recursive=True):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)


def _run_pipeline(
    survey,
    update_test_version,
    instrument_names,
    aper_diams,
    forced_phot_stacked_band_data_from_arr,
    update,
):
    from astroquery.gaia import Gaia

    mp = pytest.MonkeyPatch()
    mp.setattr(Gaia, "launch_job_async", _mock_gaia_launch_job_async)
    try:
        return Data.pipeline(
            survey,
            update_test_version,
            instrument_names,
            aper_diams=aper_diams,
            forced_phot_band=forced_phot_stacked_band_data_from_arr,
            version_to_dir_dict={
                update_test_version.split("_")[0]: update_test_version.split(
                    "_"
                )[0]
            },
            im_str=["test"],
            update=update,
        )
    finally:
        mp.undo()


@pytest.mark.requires_data
@pytest.mark.slow
def test_update_false_still_builds_catalogue_from_scratch(
    survey,
    update_test_version,
    instrument_names,
    aper_diams,
    forced_phot_stacked_band_data_from_arr,
):
    # nothing exists on disk yet for this version -- update=False must not
    # skip the initial build
    data = _run_pipeline(
        survey,
        update_test_version,
        instrument_names,
        aper_diams,
        forced_phot_stacked_band_data_from_arr,
        update=False,
    )
    assert Path(data.phot_cat_path).is_file()
    tab = Table.read(data.phot_cat_path, hdu="OBJECTS")
    assert len(tab) > 0
    assert all(
        f"loc_depth_{band_data.filt_name}" in tab.colnames
        for band_data in data
    )


@pytest.mark.requires_data
@pytest.mark.slow
def test_update_false_skips_rewrite_once_built(
    survey,
    update_test_version,
    instrument_names,
    aper_diams,
    forced_phot_stacked_band_data_from_arr,
):
    data1 = _run_pipeline(
        survey,
        update_test_version,
        instrument_names,
        aper_diams,
        forced_phot_stacked_band_data_from_arr,
        update=False,
    )
    cat_path = data1.phot_cat_path
    mtime_after_build = Path(cat_path).stat().st_mtime

    # a second, independent Data object for the same survey/version
    # simulates a fresh script run against an already-built catalogue
    _run_pipeline(
        survey,
        update_test_version,
        instrument_names,
        aper_diams,
        forced_phot_stacked_band_data_from_arr,
        update=False,
    )
    mtime_after_repeat = Path(cat_path).stat().st_mtime
    assert mtime_after_repeat == mtime_after_build

    # explicitly requesting update=True must still force a rewrite
    _run_pipeline(
        survey,
        update_test_version,
        instrument_names,
        aper_diams,
        forced_phot_stacked_band_data_from_arr,
        update=True,
    )
    mtime_after_forced_update = Path(cat_path).stat().st_mtime
    assert mtime_after_forced_update > mtime_after_repeat

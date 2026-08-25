import os
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest

from galfind.utils import utils
from galfind.utils.exceptions import InvalidOptionError


def test_get_data_dir_default_pix_scale():
    result = utils.get_data_dir("/data", "MySurvey", "v1", ["NIRCam"])
    assert list(result) == ["/data/jwst/MySurvey/NIRCam/v1/30mas"]


def test_get_data_dir_custom_pix_scale():
    result = utils.get_data_dir(
        "/data", "MySurvey", "v1", ["NIRCam"], pix_scale=0.06 * u.arcsec
    )
    assert list(result) == ["/data/jwst/MySurvey/NIRCam/v1/60mas"]


def test_get_data_dir_version_to_dir_dict():
    result = utils.get_data_dir(
        "/data",
        "MySurvey",
        "v1",
        ["NIRCam"],
        version_to_dir_dict={"v1": "other_dir"},
    )
    assert list(result) == ["/data/jwst/MySurvey/NIRCam/other_dir/30mas"]


def test_get_data_dir_unrecognised_version_raises():
    with pytest.raises(InvalidOptionError):
        utils.get_data_dir(
            "/data",
            "MySurvey",
            "bad_version",
            ["NIRCam"],
            version_to_dir_dict={"v1": "other_dir"},
        )


def test_get_data_dir_unrecognised_instrument_raises():
    with pytest.raises(InvalidOptionError):
        utils.get_data_dir("/data", "MySurvey", "v1", ["BogusInstrument"])


def test_get_cat_dir_joins_instrument_names():
    result = utils.get_cat_dir(
        "/work", "MySurvey", "v1", ["NIRCam", "ACS_WFC"]
    )
    assert list(result) == ["/work/Catalogues/v1/NIRCam+ACS_WFC/MySurvey"]


def test_get_depth_dir_one_per_instrument():
    result = utils.get_depth_dir(
        "/work", "MySurvey", "v1", ["NIRCam", "ACS_WFC"]
    )
    assert list(result) == [
        "/work/Depths/NIRCam/v1/MySurvey",
        "/work/Depths/ACS_WFC/v1/MySurvey",
    ]


def test_get_eazy_dir_returns_input_and_output():
    result = utils.get_eazy_dir("/work", "MySurvey", "v1", ["NIRCam"])
    assert list(result) == [
        "/work/EAZY/input/NIRCam/v1/MySurvey",
        "/work/EAZY/output/NIRCam/v1/MySurvey",
    ]


def test_get_mask_dir():
    result = utils.get_mask_dir("/work", "MySurvey")
    assert list(result) == ["/work/Masks/MySurvey"]


def test_get_sex_dir():
    result = utils.get_sex_dir("/work", "MySurvey", "v1", ["NIRCam"])
    assert list(result) == ["/work/SExtractor/NIRCam/v1/MySurvey"]


def test_get_stacked_images_dir():
    result = utils.get_stacked_images_dir(
        "/work", "MySurvey", "v1", ["NIRCam"]
    )
    assert list(result) == ["/work/Stacked_Images/v1/NIRCam/MySurvey"]


@pytest.mark.parametrize(
    "keyword,expected_func",
    [
        ("Data", "get_data_dir"),
        ("Catalogues", "get_cat_dir"),
        ("Depths", "get_depth_dir"),
        ("EAZY", "get_eazy_dir"),
        ("Masks", "get_mask_dir"),
        ("SExtractor", "get_sex_dir"),
        ("Stacked_Images", "get_stacked_images_dir"),
    ],
)
def test_find_target_dir_dispatches_to_correct_function(
    keyword, expected_func
):
    with patch.object(utils, expected_func) as mocked:
        mocked.return_value = np.array(["dummy"])
        result = utils.find_target_dir(
            "/work", "MySurvey", "v1", ["NIRCam"], keyword
        )
        mocked.assert_called_once()
        assert list(result) == ["dummy"]


def test_find_target_dir_unrecognised_keyword_raises():
    with pytest.raises(ValueError):
        utils.find_target_dir("/work", "MySurvey", "v1", ["NIRCam"], "Bogus")


def test_is_up_to_date_true_when_size_and_mtime_match(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("content")
    stat = f.stat()
    assert utils._is_up_to_date(str(f), stat.st_size, stat.st_mtime) is True


def test_is_up_to_date_false_when_size_differs(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("content")
    stat = f.stat()
    assert (
        utils._is_up_to_date(str(f), stat.st_size + 1, stat.st_mtime) is False
    )


def test_is_up_to_date_false_when_dest_older(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("content")
    stat = f.stat()
    assert (
        utils._is_up_to_date(str(f), stat.st_size, stat.st_mtime - 1000)
        is False
    )


def test_local_copy_copies_new_file(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dest = tmp_path / "nested" / "dest.txt"
    result = utils._local_copy(str(src), str(dest))
    assert result is True
    assert dest.read_text() == "hello"


def test_local_copy_skips_when_up_to_date(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dest = tmp_path / "dest.txt"
    utils._local_copy(str(src), str(dest))
    result = utils._local_copy(str(src), str(dest))
    assert result is False


def test_local_copy_force_recopies(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dest = tmp_path / "dest.txt"
    utils._local_copy(str(src), str(dest))
    result = utils._local_copy(str(src), str(dest), force=True)
    assert result is True


def test_local_copy_recopies_when_content_changed(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dest = tmp_path / "dest.txt"
    utils._local_copy(str(src), str(dest))
    src.write_text("hello world!")
    result = utils._local_copy(str(src), str(dest))
    assert result is True
    assert dest.read_text() == "hello world!"


def test_remote_copy_creates_parent_dir_and_puts_file(tmp_path):
    ssh_client = MagicMock()
    channel = MagicMock()
    channel.recv_exit_status.return_value = 0
    stdout = MagicMock()
    stdout.channel = channel
    ssh_client.exec_command.return_value = (MagicMock(), stdout, MagicMock())

    sftp_client = MagicMock()
    sftp_client.stat.side_effect = FileNotFoundError()

    src = tmp_path / "src.txt"
    src.write_text("hello")

    result = utils._remote_copy(
        ssh_client, sftp_client, str(src), "/remote/dest.txt"
    )
    assert result is True
    sftp_client.put.assert_called_once_with(str(src), "/remote/dest.txt")
    sftp_client.utime.assert_called_once()


def test_remote_copy_skips_when_up_to_date(tmp_path):
    ssh_client = MagicMock()
    channel = MagicMock()
    channel.recv_exit_status.return_value = 0
    stdout = MagicMock()
    stdout.channel = channel
    ssh_client.exec_command.return_value = (MagicMock(), stdout, MagicMock())

    src = tmp_path / "src.txt"
    src.write_text("hello")
    src_stat = src.stat()

    sftp_client = MagicMock()
    remote_stat = MagicMock()
    remote_stat.st_size = src_stat.st_size
    remote_stat.st_mtime = src_stat.st_mtime
    sftp_client.stat.return_value = remote_stat

    result = utils._remote_copy(
        ssh_client, sftp_client, str(src), "/remote/dest.txt"
    )
    assert result is False
    sftp_client.put.assert_not_called()


def test_symlink_and_unlink_round_trip(tmp_path):
    target_root = tmp_path / "target"
    symlink_root = tmp_path / "symlink"
    survey = "MySurvey"
    version = "v1"
    instrument_names = ["NIRCam"]

    mask_dir = target_root / "Masks" / survey
    mask_dir.mkdir(parents=True)
    real_file = mask_dir / "mask.fits"
    real_file.write_text("mask data")

    utils.symlink(
        str(target_root),
        str(symlink_root),
        survey,
        version,
        instrument_names,
        ["Masks"],
    )

    linked_file = symlink_root / "Masks" / survey / "mask.fits"
    assert linked_file.is_symlink()
    assert os.path.realpath(linked_file) == os.path.realpath(real_file)
    assert linked_file.read_text() == "mask data"

    utils.unlink(
        str(symlink_root), survey, version, instrument_names, ["Masks"]
    )
    assert not linked_file.exists()
    assert not linked_file.is_symlink()
    assert real_file.exists()


def test_unlink_leaves_regular_files_untouched(tmp_path):
    target_root = tmp_path / "target"
    survey = "MySurvey"
    version = "v1"
    instrument_names = ["NIRCam"]

    mask_dir = target_root / "Masks" / survey
    mask_dir.mkdir(parents=True)
    real_file = mask_dir / "mask.fits"
    real_file.write_text("mask data")

    utils.unlink(
        str(target_root), survey, version, instrument_names, ["Masks"]
    )
    assert real_file.exists()


def test_remote_copy_raises_when_mkdir_fails(tmp_path):
    ssh_client = MagicMock()
    channel = MagicMock()
    channel.recv_exit_status.return_value = 1
    stdout = MagicMock()
    stdout.channel = channel
    stderr = MagicMock()
    stderr.read.return_value = b"permission denied"
    ssh_client.exec_command.return_value = (MagicMock(), stdout, stderr)

    sftp_client = MagicMock()
    src = tmp_path / "src.txt"
    src.write_text("hello")

    with pytest.raises(RuntimeError):
        utils._remote_copy(
            ssh_client, sftp_client, str(src), "/remote/dest.txt"
        )

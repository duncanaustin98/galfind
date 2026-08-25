import logging
import os
import warnings
from unittest.mock import MagicMock, patch

import pytest

from galfind.utils import decorators
from galfind.utils.exceptions import GalfindError


def test_run_in_dir_executes_inside_target_directory(tmp_path):
    target = tmp_path / "workdir"
    original_cwd = os.getcwd()

    @decorators.run_in_dir(str(target))
    def get_cwd():
        return os.getcwd()

    result = get_cwd()
    assert os.path.realpath(result) == os.path.realpath(str(target))
    assert os.getcwd() == original_cwd


def test_run_in_dir_creates_missing_directory(tmp_path):
    target = tmp_path / "does_not_exist_yet"
    assert not target.exists()

    @decorators.run_in_dir(str(target))
    def noop():
        return True

    assert noop() is True
    assert target.is_dir()


def test_run_in_dir_restores_cwd_on_exception(tmp_path):
    target = tmp_path / "workdir"
    original_cwd = os.getcwd()

    @decorators.run_in_dir(str(target))
    def raises():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        raises()
    assert os.getcwd() == original_cwd


def test_run_in_dir_passes_args_and_kwargs(tmp_path):
    target = tmp_path / "workdir"

    @decorators.run_in_dir(str(target))
    def add(a, b, c=0):
        return a + b + c

    assert add(1, 2, c=3) == 6


def test_run_in_self_dir_uses_instance_attribute(tmp_path):
    original_cwd = os.getcwd()

    class Foo:
        def __init__(self, workdir):
            self.workdir = workdir

        @decorators.run_in_self_dir(lambda self: self.workdir)
        def get_cwd(self):
            return os.getcwd()

    foo = Foo(str(tmp_path / "instance_dir"))
    result = foo.get_cwd()
    assert os.path.realpath(result) == os.path.realpath(foo.workdir)
    assert os.getcwd() == original_cwd


def test_run_in_self_dir_creates_missing_directory(tmp_path):
    class Foo:
        def __init__(self, workdir):
            self.workdir = workdir

        @decorators.run_in_self_dir(lambda self: self.workdir)
        def noop(self):
            return True

    foo = Foo(str(tmp_path / "brand_new_dir"))
    assert not os.path.exists(foo.workdir)
    assert foo.noop() is True
    assert os.path.isdir(foo.workdir)


def test_log_time_returns_wrapped_function_result():
    @decorators.log_time(logging.INFO)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_log_time_logs_start_and_completion():
    @decorators.log_time(logging.WARNING)
    def noop():
        return None

    with patch.object(decorators, "galfind_logger") as mock_logger:
        noop()
        mock_logger.info.assert_called_once()
        mock_logger.log.assert_called_once()
        assert mock_logger.log.call_args[0][0] == logging.WARNING


def test_hour_timer_returns_wrapped_function_result(capsys):
    @decorators.hour_timer
    def multiply(a, b):
        return a * b

    result = multiply(3, 4)
    assert result == 12
    captured = capsys.readouterr()
    assert "multiply" in captured.out
    assert "executed in" in captured.out


def test_ignore_warnings_suppresses_warnings():
    @decorators.ignore_warnings
    def emit_warning():
        warnings.warn("this should be suppressed", UserWarning)
        return "done"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = emit_warning()
        assert result == "done"
        assert len(caught) == 0


def test_ignore_warnings_still_returns_value_and_propagates_exceptions():
    @decorators.ignore_warnings
    def raises():
        raise KeyError("boom")

    with pytest.raises(KeyError):
        raises()


def test_n_cores_calls_wrapped_function_unmodified():
    @decorators.n_cores(4)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_email_update_sends_start_and_end_emails():
    mock_smtp_instance = MagicMock()
    with patch.object(
        decorators.yagmail, "SMTP", return_value=mock_smtp_instance
    ) as mock_smtp:

        @decorators.email_update(to="test@example.com")
        def noop():
            return "result"

        result = noop()
        assert result == "result"
        mock_smtp.assert_called_once()
        assert mock_smtp_instance.send.call_count == 2


def test_email_update_sends_terminate_email_and_reraises_on_exception():
    mock_smtp_instance = MagicMock()
    with patch.object(
        decorators.yagmail, "SMTP", return_value=mock_smtp_instance
    ):

        @decorators.email_update(to="test@example.com")
        def raises():
            raise ValueError("original error")

        with pytest.raises(GalfindError):
            raises()
        assert mock_smtp_instance.send.call_count == 2

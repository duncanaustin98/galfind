import logging

import pytest

from galfind.utils import exceptions
from galfind.utils.exceptions import (
    AbstractMethodError,
    EmptyCatalogueError,
    ExternalToolError,
    GalfindError,
    GalfindTypeError,
    IncompatibleKwargsError,
    InvalidOptionError,
    InvalidUnitError,
    LengthMismatchError,
    MissingDataError,
    MissingFileError,
    MissingKeyError,
    RangeError,
)

# every (subclass, extra builtin base) pair, mirroring the table in the
# exceptions.py module docstring
SUBCLASS_BUILTIN_PAIRS = [
    (GalfindTypeError, TypeError),
    (InvalidUnitError, ValueError),
    (InvalidOptionError, ValueError),
    (MissingKeyError, KeyError),
    (MissingDataError, GalfindError),
    (MissingFileError, FileNotFoundError),
    (LengthMismatchError, ValueError),
    (RangeError, ValueError),
    (IncompatibleKwargsError, ValueError),
    (ExternalToolError, RuntimeError),
    (AbstractMethodError, NotImplementedError),
    (EmptyCatalogueError, IndexError),
]


@pytest.mark.parametrize("cls,builtin_base", SUBCLASS_BUILTIN_PAIRS)
def test_subclass_is_instance_of_galfind_error_and_builtin(cls, builtin_base):
    exc = cls("something went wrong")
    assert isinstance(exc, GalfindError)
    assert isinstance(exc, builtin_base)


@pytest.mark.parametrize("cls,_builtin_base", SUBCLASS_BUILTIN_PAIRS)
def test_message_is_preserved_on_the_exception_object(cls, _builtin_base):
    # this is the actual bug being fixed: `assert cond, logger.critical(msg)`
    # raises an AssertionError with NO message at all, since
    # `logger.critical()` returns None. every subclass here must carry the
    # real message on the exception itself.
    exc = cls("a very specific, actionable message")
    assert str(exc) == "a very specific, actionable message"
    assert exc.args[0] == "a very specific, actionable message"


def test_message_str_is_not_repr_quoted_for_keyerror_mixin():
    # bare KeyError("foo") stringifies as "'foo'" (repr-quoted); GalfindError
    # must override __str__ so MissingKeyError doesn't inherit that quirk.
    assert str(KeyError("foo")) == "'foo'"
    assert str(MissingKeyError("foo")) == "foo"


def test_can_be_raised_and_caught_narrowly_and_broadly():
    with pytest.raises(InvalidOptionError):
        raise InvalidOptionError("bad option")
    with pytest.raises(ValueError):
        raise InvalidOptionError("bad option")
    with pytest.raises(GalfindError):
        raise InvalidOptionError("bad option")


def test_construction_logs_at_the_requested_level(caplog):
    with caplog.at_level(logging.WARNING, logger="galfind"):
        GalfindError("careful now", log_level="warning")
    assert any(
        record.levelno == logging.WARNING and "careful now" in record.message
        for record in caplog.records
    )


def test_construction_defaults_to_error_level(caplog):
    with caplog.at_level(logging.ERROR, logger="galfind"):
        GalfindError("default level message")
    assert any(
        record.levelno == logging.ERROR
        and "default level message" in record.message
        for record in caplog.records
    )


class TestAbstractMethodError:
    def test_custom_message_is_used_verbatim(self):
        exc = AbstractMethodError("custom explanation")
        assert str(exc) == "custom explanation"

    def test_default_message_names_class_and_method(self):
        class Base:
            def compute(self):
                raise AbstractMethodError()

        class Concrete(Base):
            pass

        with pytest.raises(AbstractMethodError) as excinfo:
            Concrete().compute()
        assert "Concrete" in str(excinfo.value)
        assert "compute" in str(excinfo.value)


def test_module_exports_match_public_api():
    for name in exceptions.__all__:
        assert hasattr(exceptions, name)
        assert issubclass(getattr(exceptions, name), Exception)

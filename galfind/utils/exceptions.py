"""Custom exception hierarchy for galfind.

Every validation failure in galfind used to be raised as a bare
``assert condition, galfind_logger.critical(err_message)``. This is a bug, not
just a style choice: ``Logger.critical(msg)`` returns ``None``, so the
``AssertionError`` actually raised when the assertion fails carries **no
message at all** -- ``err_message`` only ever reaches the log handler, never
the exception object seen by calling code or by ``pytest.raises``. The
resulting failures were also all indistinguishable ``AssertionError``\\ s
regardless of what actually went wrong (a bad unit, a missing key, an
out-of-range value, ...).

This module defines a small, shared hierarchy instead. Every subclass here:

* logs its message via the shared ``galfind`` logger *and* attaches it to the
  exception object itself, fixing the "message never reaches the caller" bug.
* also inherits from the built-in exception type its category conceptually
  matches (e.g. `InvalidOptionError` is also a `ValueError`), so existing code
  that catches broad built-in types, and any `pytest.raises(ValueError)` /
  `pytest.raises(TypeError)` style test, keeps working unchanged. New code can
  either catch narrowly (``except InvalidOptionError``) or broadly
  (``except GalfindError``).

Use the most specific subclass that fits; fall back to `GalfindError` directly
only if nothing below fits the failure being reported.

Examples
--------
>>> from galfind.utils.exceptions import InvalidOptionError
>>> allowed = ["UV", "optical"]
>>> value = "xray"
>>> if value not in allowed:
...     raise InvalidOptionError(
...         f"Unrecognised 'band' = {value!r} passed to "
...         f"Foo_Calculator; must be one of {allowed}."
...     )
Traceback (most recent call last):
    ...
galfind.utils.exceptions.InvalidOptionError: Unrecognised 'band' = 'xray'
passed to Foo_Calculator; must be one of ['UV', 'optical'].
"""

from __future__ import annotations

import inspect
import logging

# NOTE: intentionally *not* `from . import galfind_logger` / `from .. import
# galfind_logger`. `galfind/__init__.py` creates the shared logger via
# `logging.getLogger(__name__)` where `__name__ == "galfind"` at that point;
# `logging.getLogger` is a registry keyed by name, so asking for "galfind"
# here returns that exact same `Logger` instance without importing anything
# from the package `__init__` (which is still being built the first time
# this module is imported, and would otherwise risk a circular import).
_galfind_logger = logging.getLogger("galfind")

__all__ = [
    "GalfindError",
    "GalfindTypeError",
    "InvalidUnitError",
    "InvalidOptionError",
    "MissingKeyError",
    "MissingDataError",
    "MissingFileError",
    "LengthMismatchError",
    "RangeError",
    "IncompatibleKwargsError",
    "ExternalToolError",
    "AbstractMethodError",
    "EmptyCatalogueError",
]


class GalfindError(Exception):
    """Base class for every custom galfind exception.

    Parameters
    ----------
    message : `str`
        Human-readable, actionable description of what went wrong -- should
        name the offending parameter/value and what was expected.
    log_level : `str`, optional
        Level to log ``message`` at via the shared ``"galfind"`` logger
        before raising (one of ``"debug"``, ``"info"``, ``"warning"``,
        ``"error"``, ``"critical"``). Defaults to ``"error"``.

    Notes
    -----
    Subclasses that also inherit from `KeyError` would otherwise pick up
    `KeyError.__str__`'s behaviour of `repr`-quoting a single string argument
    (e.g. ``str(KeyError("foo"))`` is ``"'foo'"``, not ``"foo"``). Because
    `GalfindError` is earlier in the MRO than `KeyError` for those subclasses
    and defines `__str__` itself, every subclass here (whatever it mixes in)
    reports the plain message.
    """

    def __init__(self, message: str, *, log_level: str = "error"):
        log_fn = getattr(_galfind_logger, log_level, _galfind_logger.error)
        log_fn(message)
        super().__init__(message)

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class GalfindTypeError(GalfindError, TypeError):
    """Raised when an argument or attribute is not of the required type.

    Covers ``isinstance`` check failures -- e.g. an `aper_diam` that isn't an
    `astropy.units.Quantity`, or a `SED_fit_label` that isn't a `str`.
    """


class InvalidUnitError(GalfindError, ValueError):
    """Raised when an `astropy.units.Quantity` has the wrong unit, the wrong
    physical type (e.g. a wavelength expected but a dimensionless value or a
    frequency given), or is inconsistent with other quantities it must share
    units with (e.g. mismatched units across an array of measurements).
    """


class InvalidOptionError(GalfindError, ValueError):
    """Raised when a value is not one of a fixed set of allowed options.

    Covers string-enum-style kwargs (``model in self._available_models``,
    ``SFR_conv in funcs.SFR_conversions``, ``method in ["rms_err", "wht"]``,
    ...). The message should always include both the received value and the
    allowed set.
    """


class MissingKeyError(GalfindError, KeyError):
    """Raised when a required key is absent from a plain `dict` / `kwargs`
    (e.g. a required calibration or config key). For data that is *derived*
    on a `Galaxy`/`Catalogue` rather than supplied by the caller (an aperture
    photometry or SED fit that hasn't been run yet), use `MissingDataError`
    instead -- the fix for that is "run the missing step first", not "pass a
    different keyword argument".
    """


class MissingDataError(GalfindError):
    """Raised when a `Galaxy`/`Catalogue`/`Photometry` object is missing
    derived data a property calculator or selector needs -- e.g. the
    requested `aper_diam` has no aperture photometry loaded, or the
    requested `SED_fit_label` has not been fit yet. The message should name
    the missing step so the caller knows what to run first.
    """


class MissingFileError(GalfindError, FileNotFoundError):
    """Raised when a path expected to exist on disk (a config file, a
    catalogue, an external tool's output file) is missing.
    """


class LengthMismatchError(GalfindError, ValueError):
    """Raised when two or more arrays/lists/tables that are required to
    correspond element-for-element have different lengths or shapes.
    """


class RangeError(GalfindError, ValueError):
    """Raised when a numeric value violates a required bound or ordering
    constraint (e.g. must be positive, must lie within [0, 1], a lower limit
    that isn't actually lower than the upper limit).
    """


class IncompatibleKwargsError(GalfindError, ValueError):
    """Raised when two or more keyword arguments are mutually exclusive, or
    co-dependent, and were given in a way that violates that relationship
    (e.g. both given when only one is allowed, or one given without a
    required companion).
    """


class ExternalToolError(GalfindError, RuntimeError):
    """Raised when an external tool (SExtractor, EAZY, LePhare, GALFIT,
    PySersic, ...) is misconfigured, fails to run, or produces output that
    doesn't match the contract galfind expects from it (missing/malformed
    output file, unexpected header keyword, ...).
    """


class AbstractMethodError(GalfindError, NotImplementedError):
    """Raised from a base-class stub that concrete subclasses are required
    to override.

    Building the message from `inspect` means call sites don't need to
    hand-roll the class/method name themselves::

        def _kwarg_assertions(self):
            raise AbstractMethodError()
    """

    def __init__(
        self, message: str | None = None, *, log_level: str = "error"
    ):
        if message is None:
            frame = inspect.currentframe()
            outer = frame.f_back if frame is not None else None
            method_name = (
                outer.f_code.co_name if outer is not None else "this method"
            )
            self_obj = (
                outer.f_locals.get("self") if outer is not None else None
            )
            cls_name = (
                type(self_obj).__name__
                if self_obj is not None
                else "This class"
            )
            message = f"{cls_name} must implement '{method_name}()'."
        super().__init__(message, log_level=log_level)


class EmptyCatalogueError(GalfindError, IndexError):
    """Raised when an operation that requires at least one `Galaxy` is
    attempted on an empty `Catalogue` (e.g. indexing or iterating).
    """

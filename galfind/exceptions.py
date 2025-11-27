"""
Custom exception classes for the galfind package.

This module defines a hierarchy of custom exceptions to provide better error
tracing and more informative error messages throughout the codebase.
"""

import logging

# Get the galfind logger
_logger = logging.getLogger("galfind")


class GalfindError(Exception):
    """Base exception class for all galfind-related errors.

    Automatically logs the error message at CRITICAL level when raised.
    """

    def __init__(self, message: str = "", log_level: int = logging.CRITICAL):
        """Initialize the exception and log the message.

        Parameters
        ----------
        message : str
            The error message.
        log_level : int
            The logging level to use (default: logging.CRITICAL).
        """
        super().__init__(message)
        if message:
            _logger.log(log_level, message)


class CatalogueError(GalfindError):
    """Exception raised for errors related to catalogue operations."""

    pass


class DataError(GalfindError):
    """Exception raised for errors related to data loading or processing."""

    pass


class ConfigurationError(GalfindError):
    """Exception raised for configuration-related issues."""

    pass


class SelectorError(GalfindError):
    """Exception raised for errors in selector/selection operations."""

    pass


class SEDFittingError(GalfindError):
    """Exception raised for SED fitting related errors."""

    pass


class MissingDataError(GalfindError):
    """Exception raised when required data or attributes are missing."""

    pass


class InvalidArgumentError(GalfindError):
    """Exception raised when invalid arguments are provided."""

    pass


class PlottingError(GalfindError):
    """Exception raised for plotting-related errors."""

    pass


class MCMCError(GalfindError):
    """Exception raised for MCMC fitting related errors."""

    pass

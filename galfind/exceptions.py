"""
Custom exception classes for the galfind package.

This module defines a hierarchy of custom exceptions to provide better error
tracing and more informative error messages throughout the codebase.
"""


class GalfindError(Exception):
    """Base exception class for all galfind-related errors."""

    pass


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

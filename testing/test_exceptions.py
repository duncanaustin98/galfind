"""
Unit tests for the galfind.exceptions module.

Tests all custom exception classes to ensure proper hierarchy,
error messaging, and logging behavior.
"""

import logging
import pytest
import sys
import os

# Add galfind source directory to path to import exceptions directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'galfind'))

from exceptions import (
    GalfindError,
    CatalogueError,
    DataError,
    ConfigurationError,
    SelectorError,
    SEDFittingError,
    MissingDataError,
    InvalidArgumentError,
    PlottingError,
    MCMCError,
)


class TestGalfindError:
    """Tests for the base GalfindError exception class."""

    def test_galfind_error_is_exception(self):
        """GalfindError should inherit from Exception."""
        assert issubclass(GalfindError, Exception)

    def test_galfind_error_with_message(self):
        """GalfindError should store and return the error message."""
        msg = "Test error message"
        error = GalfindError(msg)
        assert str(error) == msg

    def test_galfind_error_empty_message(self):
        """GalfindError should handle empty message."""
        error = GalfindError()
        assert str(error) == ""

    def test_galfind_error_can_be_raised(self):
        """GalfindError should be raiseable."""
        with pytest.raises(GalfindError):
            raise GalfindError("Test error")

    def test_galfind_error_can_be_caught(self):
        """GalfindError should be catchable."""
        try:
            raise GalfindError("Test error")
        except GalfindError as e:
            assert str(e) == "Test error"

    def test_galfind_error_logs_message(self, caplog):
        """GalfindError should log the message at CRITICAL level."""
        with caplog.at_level(logging.CRITICAL, logger="galfind"):
            try:
                raise GalfindError("Test logging message")
            except GalfindError:
                pass
        assert "Test logging message" in caplog.text

    def test_galfind_error_custom_log_level(self, caplog):
        """GalfindError should support custom log levels."""
        with caplog.at_level(logging.WARNING, logger="galfind"):
            try:
                raise GalfindError("Warning message", log_level=logging.WARNING)
            except GalfindError:
                pass
        assert "Warning message" in caplog.text


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            CatalogueError,
            DataError,
            ConfigurationError,
            SelectorError,
            SEDFittingError,
            MissingDataError,
            InvalidArgumentError,
            PlottingError,
            MCMCError,
        ],
    )
    def test_all_exceptions_inherit_from_galfind_error(self, exception_class):
        """All custom exceptions should inherit from GalfindError."""
        assert issubclass(exception_class, GalfindError)

    @pytest.mark.parametrize(
        "exception_class",
        [
            CatalogueError,
            DataError,
            ConfigurationError,
            SelectorError,
            SEDFittingError,
            MissingDataError,
            InvalidArgumentError,
            PlottingError,
            MCMCError,
        ],
    )
    def test_all_exceptions_inherit_from_exception(self, exception_class):
        """All custom exceptions should also inherit from Exception."""
        assert issubclass(exception_class, Exception)

    @pytest.mark.parametrize(
        "exception_class",
        [
            CatalogueError,
            DataError,
            ConfigurationError,
            SelectorError,
            SEDFittingError,
            MissingDataError,
            InvalidArgumentError,
            PlottingError,
            MCMCError,
        ],
    )
    def test_all_exceptions_catchable_as_galfind_error(self, exception_class):
        """All custom exceptions should be catchable as GalfindError."""
        with pytest.raises(GalfindError):
            raise exception_class("Test message")


class TestCatalogueError:
    """Tests for CatalogueError exception."""

    def test_catalogue_error_with_message(self):
        """CatalogueError should store the error message."""
        error = CatalogueError("Catalogue not found")
        assert str(error) == "Catalogue not found"

    def test_catalogue_error_can_be_raised(self):
        """CatalogueError should be raiseable."""
        with pytest.raises(CatalogueError):
            raise CatalogueError("Test catalogue error")


class TestDataError:
    """Tests for DataError exception."""

    def test_data_error_with_message(self):
        """DataError should store the error message."""
        error = DataError("Data loading failed")
        assert str(error) == "Data loading failed"

    def test_data_error_can_be_raised(self):
        """DataError should be raiseable."""
        with pytest.raises(DataError):
            raise DataError("Test data error")


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_configuration_error_with_message(self):
        """ConfigurationError should store the error message."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"

    def test_configuration_error_can_be_raised(self):
        """ConfigurationError should be raiseable."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test config error")


class TestSelectorError:
    """Tests for SelectorError exception."""

    def test_selector_error_with_message(self):
        """SelectorError should store the error message."""
        error = SelectorError("Invalid selector input")
        assert str(error) == "Invalid selector input"

    def test_selector_error_can_be_raised(self):
        """SelectorError should be raiseable."""
        with pytest.raises(SelectorError):
            raise SelectorError("Test selector error")


class TestSEDFittingError:
    """Tests for SEDFittingError exception."""

    def test_sed_fitting_error_with_message(self):
        """SEDFittingError should store the error message."""
        error = SEDFittingError("SED fitting failed")
        assert str(error) == "SED fitting failed"

    def test_sed_fitting_error_can_be_raised(self):
        """SEDFittingError should be raiseable."""
        with pytest.raises(SEDFittingError):
            raise SEDFittingError("Test SED error")


class TestMissingDataError:
    """Tests for MissingDataError exception."""

    def test_missing_data_error_with_message(self):
        """MissingDataError should store the error message."""
        error = MissingDataError("Required data missing")
        assert str(error) == "Required data missing"

    def test_missing_data_error_can_be_raised(self):
        """MissingDataError should be raiseable."""
        with pytest.raises(MissingDataError):
            raise MissingDataError("Test missing data error")


class TestInvalidArgumentError:
    """Tests for InvalidArgumentError exception."""

    def test_invalid_argument_error_with_message(self):
        """InvalidArgumentError should store the error message."""
        error = InvalidArgumentError("Invalid argument provided")
        assert str(error) == "Invalid argument provided"

    def test_invalid_argument_error_can_be_raised(self):
        """InvalidArgumentError should be raiseable."""
        with pytest.raises(InvalidArgumentError):
            raise InvalidArgumentError("Test invalid argument error")


class TestPlottingError:
    """Tests for PlottingError exception."""

    def test_plotting_error_with_message(self):
        """PlottingError should store the error message."""
        error = PlottingError("Plotting failed")
        assert str(error) == "Plotting failed"

    def test_plotting_error_can_be_raised(self):
        """PlottingError should be raiseable."""
        with pytest.raises(PlottingError):
            raise PlottingError("Test plotting error")


class TestMCMCError:
    """Tests for MCMCError exception."""

    def test_mcmc_error_with_message(self):
        """MCMCError should store the error message."""
        error = MCMCError("MCMC fitting failed")
        assert str(error) == "MCMC fitting failed"

    def test_mcmc_error_can_be_raised(self):
        """MCMCError should be raiseable."""
        with pytest.raises(MCMCError):
            raise MCMCError("Test MCMC error")


class TestExceptionChaining:
    """Tests for exception chaining functionality."""

    def test_exception_can_chain_from_other_exception(self):
        """Custom exceptions should support exception chaining with 'from'."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DataError("Wrapped error") from e
        except DataError as e:
            assert str(e) == "Wrapped error"
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"

    def test_exception_chaining_preserves_traceback(self):
        """Exception chaining should preserve the original traceback."""
        try:
            try:
                raise KeyError("Missing key")
            except KeyError as e:
                raise CatalogueError("Catalogue operation failed") from e
        except CatalogueError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, KeyError)

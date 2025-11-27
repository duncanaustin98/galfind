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


class TestInvalidInputs:
    """Tests for invalid/stupid inputs that should properly fail."""

    # Test that exceptions handle various invalid message types gracefully
    def test_exception_with_none_message(self):
        """Exceptions should handle None as message."""
        error = GalfindError(None)
        assert str(error) == "None"

    def test_exception_with_integer_message(self):
        """Exceptions should handle integer as message."""
        error = GalfindError(123)
        assert str(error) == "123"

    def test_exception_with_list_message(self):
        """Exceptions should handle list as message."""
        error = GalfindError([1, 2, 3])
        assert str(error) == "[1, 2, 3]"

    def test_exception_with_dict_message(self):
        """Exceptions should handle dict as message."""
        error = GalfindError({"key": "value"})
        assert "key" in str(error)

    def test_exception_with_empty_string(self):
        """Exceptions should handle empty string as message."""
        error = GalfindError("")
        assert str(error) == ""

    def test_exception_with_whitespace_only(self):
        """Exceptions should handle whitespace-only message."""
        error = GalfindError("   ")
        assert str(error) == "   "

    def test_exception_with_special_characters(self):
        """Exceptions should handle special characters in message."""
        special_msg = "Error: !@#$%^&*()_+-={}[]|\\:\";<>?,./~`"
        error = GalfindError(special_msg)
        assert str(error) == special_msg

    def test_exception_with_unicode_characters(self):
        """Exceptions should handle unicode characters in message."""
        unicode_msg = "Error: 你好世界 🚀 αβγδ"
        error = GalfindError(unicode_msg)
        assert str(error) == unicode_msg

    def test_exception_with_very_long_message(self):
        """Exceptions should handle very long messages."""
        long_msg = "A" * 10000
        error = GalfindError(long_msg)
        assert len(str(error)) == 10000

    def test_exception_with_newlines(self):
        """Exceptions should handle messages with newlines."""
        multiline_msg = "Line 1\nLine 2\nLine 3"
        error = GalfindError(multiline_msg)
        assert str(error) == multiline_msg

    def test_exception_with_tabs(self):
        """Exceptions should handle messages with tabs."""
        tab_msg = "Col1\tCol2\tCol3"
        error = GalfindError(tab_msg)
        assert str(error) == tab_msg

    # Test invalid log levels
    def test_exception_with_invalid_log_level_string(self):
        """Exceptions should handle invalid log level (string instead of int)."""
        # This should not crash, even with invalid log level
        try:
            error = GalfindError("Test", log_level="INVALID")
        except TypeError:
            # Expected - log_level should be an int
            pass

    def test_exception_with_negative_log_level(self):
        """Exceptions should handle negative log level."""
        # Negative log levels are technically valid in logging
        error = GalfindError("Test", log_level=-1)
        assert str(error) == "Test"

    def test_exception_with_zero_log_level(self):
        """Exceptions should handle zero log level (NOTSET)."""
        error = GalfindError("Test", log_level=0)
        assert str(error) == "Test"

    def test_exception_with_very_high_log_level(self):
        """Exceptions should handle very high log level."""
        error = GalfindError("Test", log_level=1000)
        assert str(error) == "Test"

    # Test raising with various invalid states
    def test_raise_exception_with_format_string(self):
        """Exceptions should handle format string placeholders."""
        with pytest.raises(GalfindError) as exc_info:
            raise GalfindError("Error: %s %d {}")
        assert "%s %d {}" in str(exc_info.value)

    def test_exception_preserves_type_after_reraise(self):
        """Exceptions should preserve their type after being re-raised."""
        try:
            try:
                raise SelectorError("Original")
            except SelectorError:
                raise
        except SelectorError as e:
            assert str(e) == "Original"
            assert type(e).__name__ == "SelectorError"

    def test_exception_with_bytes_message(self):
        """Exceptions should handle bytes as message."""
        error = GalfindError(b"byte string")
        assert "byte string" in str(error)


class TestExceptionMessageFormatting:
    """Tests for exception message formatting with various data types."""

    def test_data_error_with_path_info(self):
        """DataError should format path information correctly."""
        error = DataError(f"File not found: /path/to/file.fits")
        assert "/path/to/file.fits" in str(error)

    def test_catalogue_error_with_id_info(self):
        """CatalogueError should format ID information correctly."""
        error = CatalogueError(f"Galaxy ID 12345 not found in catalogue")
        assert "12345" in str(error)

    def test_selector_error_with_kwargs_info(self):
        """SelectorError should format kwargs information correctly."""
        kwargs = {"min_snr": 5.0, "band": "F444W"}
        error = SelectorError(f"Invalid selector inputs: {kwargs}")
        assert "min_snr" in str(error)
        assert "F444W" in str(error)

    def test_invalid_argument_with_type_info(self):
        """InvalidArgumentError should format type information correctly."""
        error = InvalidArgumentError(
            f"Expected int, got {type('string').__name__}"
        )
        assert "str" in str(error)

    def test_missing_data_error_with_attribute_info(self):
        """MissingDataError should format attribute information correctly."""
        error = MissingDataError("Missing attribute 'redshift' on Galaxy object")
        assert "redshift" in str(error)

    def test_sed_fitting_error_with_chi_squared(self):
        """SEDFittingError should format chi-squared information correctly."""
        error = SEDFittingError(f"SED fit failed: chi^2 = {999.99:.2f}")
        assert "999.99" in str(error)

    def test_mcmc_error_with_convergence_info(self):
        """MCMCError should format convergence information correctly."""
        error = MCMCError("MCMC did not converge after 10000 iterations")
        assert "10000" in str(error)

    def test_plotting_error_with_figure_info(self):
        """PlottingError should format figure information correctly."""
        error = PlottingError("Failed to create figure with shape (10, 10)")
        assert "(10, 10)" in str(error)

    def test_configuration_error_with_config_key(self):
        """ConfigurationError should format config key information correctly."""
        error = ConfigurationError("Missing config key: 'GALFIND_WORK'")
        assert "GALFIND_WORK" in str(error)


class TestExceptionUseCases:
    """Tests simulating real-world error scenarios."""

    def test_file_not_found_scenario(self):
        """Test DataError for file not found scenario."""
        filepath = "/nonexistent/path/data.fits"
        with pytest.raises(DataError) as exc_info:
            raise DataError(f"Cannot load data from {filepath}: file does not exist")
        assert filepath in str(exc_info.value)

    def test_invalid_filter_name_scenario(self):
        """Test InvalidArgumentError for invalid filter name."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            raise InvalidArgumentError("Filter 'INVALID_FILTER' not recognized")
        assert "INVALID_FILTER" in str(exc_info.value)

    def test_missing_required_column_scenario(self):
        """Test CatalogueError for missing required column."""
        with pytest.raises(CatalogueError) as exc_info:
            raise CatalogueError("Required column 'RA' not found in catalogue")
        assert "RA" in str(exc_info.value)

    def test_negative_aperture_scenario(self):
        """Test SelectorError for negative aperture diameter."""
        with pytest.raises(SelectorError) as exc_info:
            raise SelectorError("Aperture diameter cannot be negative: -0.5 arcsec")
        assert "-0.5" in str(exc_info.value)

    def test_redshift_out_of_range_scenario(self):
        """Test InvalidArgumentError for redshift out of range."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            raise InvalidArgumentError("Redshift z=-1.0 is not physical (must be >= 0)")
        assert "-1.0" in str(exc_info.value)

    def test_empty_catalogue_scenario(self):
        """Test CatalogueError for empty catalogue."""
        with pytest.raises(CatalogueError) as exc_info:
            raise CatalogueError("Cannot perform operation on empty catalogue (0 galaxies)")
        assert "0 galaxies" in str(exc_info.value)

    def test_nan_flux_scenario(self):
        """Test DataError for NaN flux values."""
        with pytest.raises(DataError) as exc_info:
            raise DataError("Flux array contains NaN values at indices [0, 5, 10]")
        assert "NaN" in str(exc_info.value)

    def test_incompatible_units_scenario(self):
        """Test InvalidArgumentError for incompatible units."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            raise InvalidArgumentError("Cannot convert 'Jy' to 'arcsec': incompatible units")
        assert "Jy" in str(exc_info.value)
        assert "arcsec" in str(exc_info.value)

    def test_sed_convergence_failure_scenario(self):
        """Test SEDFittingError for convergence failure."""
        with pytest.raises(SEDFittingError) as exc_info:
            raise SEDFittingError(
                "SED fitting failed to converge: max iterations (1000) reached"
            )
        assert "1000" in str(exc_info.value)

    def test_mcmc_walker_stuck_scenario(self):
        """Test MCMCError for stuck walkers."""
        with pytest.raises(MCMCError) as exc_info:
            raise MCMCError("MCMC walkers stuck: acceptance fraction = 0.0")
        assert "0.0" in str(exc_info.value)

    def test_plot_dimension_mismatch_scenario(self):
        """Test PlottingError for dimension mismatch."""
        with pytest.raises(PlottingError) as exc_info:
            raise PlottingError("Cannot plot: x has 100 points but y has 50 points")
        assert "100" in str(exc_info.value)
        assert "50" in str(exc_info.value)

    def test_config_type_mismatch_scenario(self):
        """Test ConfigurationError for type mismatch in config."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(
                "Config 'N_CORES' expects int, got str: 'four'"
            )
        assert "N_CORES" in str(exc_info.value)
        assert "four" in str(exc_info.value)

from __future__ import annotations

import time
from copy import deepcopy
import astropy.units as u
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde
import numpy as np
from astropy.table import Table
from typing import Callable, Union, Optional, Dict, Any, TYPE_CHECKING
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import config, galfind_logger
from . import useful_funcs_austind as funcs


class PDF:
    """Represents a 1D posterior probability distribution for a galaxy property.

    Stores the property name, the grid of x values and corresponding
    (normalised) probability density values, together with any
    additional metadata. Provides utilities to reconstruct summary
    statistics (median, percentiles, peaks), draw samples, combine
    PDFs, save/load to disk, and plot the distribution.

    Parameters
    ----------
    property_name : `str`
        Name of the galaxy property this PDF represents (e.g. ``'z'``).
    x : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
        Grid of property values at which `p_x` is defined.
    p_x : `numpy.ndarray`
        Probability density evaluated at each point in `x`.
    kwargs : `dict`, optional
        Additional metadata to store with the PDF (e.g. saved to/loaded
        from the ``.meta.npy`` file). Default is `{}`.
    normed : `bool`, optional
        Whether `p_x` is already normalised such that
        ``numpy.trapz(p_x, x) == 1``. If `False`, `p_x` is normalised
        in place upon construction. Default is `False`.

    Attributes
    ----------
    property_name : `str`
        Name of the galaxy property this PDF represents.
    x : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
        Grid of property values at which `p_x` is defined.
    p_x : `numpy.ndarray`
        Normalised probability density evaluated at each point in `x`.
    kwargs : `dict`
        Additional metadata stored with the PDF.
    input_arr : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`, optional
        Underlying sample of values the PDF was constructed from, set
        by `from_1D_arr`/`from_npy` if the PDF was built from a 1D
        array rather than directly from ``x``/``p_x``.
    save_path : `str`, optional
        Path the PDF was last saved to or loaded from, set by `save`,
        `add_save_path`, or `from_npy`.
    """

    def __init__(
        self,
        property_name,
        x,
        p_x,
        kwargs={},
        normed: bool = False,
    ):
        if not isinstance(x, tuple([u.Quantity, u.Magnitude, u.Dex])):
            breakpoint()
        # assert type(x) in [u.Quantity, u.Magnitude, u.Dex]
        self.property_name = property_name
        self.x = x
        self.kwargs = kwargs
        # normalize to np.trapz(p_x, x) == 1
        if not normed:
            p_x /= np.trapz(p_x, x.value)
        self.p_x = p_x

    def __str__(self, print_peaks=False):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = ""
        output_str += line_sep
        unit_str = (
            f"{self.x.unit}"
            if not self.x.unit == u.dimensionless_unscaled
            else "dimensionless"
        )
        output_str += f"PDF PROPERTY: {self.property_name}; UNIT: {unit_str}\n"
        output_str += band_sep
        output_str += (
            f"MEDIAN = {self.median.value:.3f}"
            + r"$_{-%.3f}^{+%.3f}$\n"
            % (self.errs.value[0], self.errs.value[1])
        )
        if print_peaks:
            for i, peak in enumerate(self.peaks):
                output_str += f"{funcs.ordinal(i + 1)} PEAK: {peak:.3f}\n"
        output_str += line_sep
        return output_str

    def __len__(self):
        if hasattr(self, "input_arr"):
            return len(self.input_arr)
        else:
            return None

    def __add__(
        self,
        other: Union[PDF, int, float, u.Quantity, u.Magnitude, u.Dex],
        name_ext: Union[str, None] = None,
        add_kwargs: dict = {},
        save: bool = False,
    ):
        if isinstance(other, (int, float, u.Quantity, u.Magnitude, u.Dex)):
            # multiply input array by other
            if hasattr(self, "input_arr"):
                old_input_arr = self.input_arr
            else:
                old_input_arr = self.draw_sample()
            new_input_arr = old_input_arr + other
            new_kwargs = {**self.kwargs, **add_kwargs}
        else:  # PDF
            # for extending length of PDF
            assert isinstance(self, type(other)), galfind_logger.critical(
                f"{type(self)=}!={type(other)=}"
            )
            assert (
                self.property_name == other.property_name
            ), galfind_logger.critical(
                f"{self.property_name=}!={other.property_name=}"
            )
            # update kwargs
            new_kwargs = {**self.kwargs, **other.kwargs, **add_kwargs}
            if hasattr(self, "input_arr"):
                self_input_arr = self.input_arr
            else:
                self_input_arr = self.draw_sample()
            if hasattr(other, "input_arr"):
                other_input_arr = other.input_arr
            else:
                other_input_arr = other.draw_sample()
            new_input_arr = np.concatenate(
                (self_input_arr, other_input_arr)
            )

        if name_ext is None:
            new_property_name = self.property_name
        else:  # type(name_ext) == str
            assert isinstance(name_ext, str), galfind_logger.critical(
                f"{name_ext=} with {type(name_ext)=} not in [str]!"
            )
            if name_ext[0] != "_":
                name_ext = f"_{name_ext}"
            new_property_name = f"{self.property_name}{name_ext}"

        if self.__class__.__name__ == "PDF":
            PDF_obj = globals()[self.__class__.__name__].from_1D_arr(
                new_property_name, new_input_arr, kwargs=new_kwargs
            )
        elif self.__class__.__name__ == "SED_fit_PDF":
            PDF_obj = globals()[self.__class__.__name__].from_1D_arr(
                new_property_name,
                new_input_arr,
                self.SED_fit_params,
                kwargs=new_kwargs,
            )
        elif self.__class__.__name__ == "Redshift_PDF":
            PDF_obj = globals()[self.__class__.__name__].from_1D_arr(
                new_input_arr, self.SED_fit_params, kwargs=new_kwargs
            )
        else:
            galfind_logger.critical(
                f"{self.__class__.__name__=} not in [PDF, SED_fit_PDF, Redshift_PDF]!"
            )
            breakpoint()
        # if chosen to save and it has a different name, save the PDF
        if (
            save
            and hasattr(self, "save_path")
            and new_property_name != self.property_name
        ):
            PDF_obj.save(
                self.save_path.replace(self.property_name, new_property_name)
            )
        return PDF_obj

    def __mul__(
        self,
        other: Union["PDF", int, float, u.Quantity, u.Magnitude, u.Dex],
        name_ext: Union[str, None] = None,
        add_kwargs: dict = {},
        save: bool = False,
    ):
        if isinstance(other, tuple(int, float, u.Quantity, u.Magnitude, u.Dex)):
            # multiply input array by other
            if hasattr(self, "input_arr"):
                old_input_arr = self.input_arr
            else:
                old_input_arr = self.draw_sample()
            new_input_arr = old_input_arr * other
            new_kwargs = {**self.kwargs, **add_kwargs}
        else:  # PDF
            # convolve the two PDFs with each other as done in Qiao's merger work
            raise NotImplementedError

        if name_ext is None:
            new_property_name = self.property_name
        else:  # type(name_ext) == str
            assert isinstance(name_ext, str), galfind_logger.critical(
                f"{name_ext=} with {type(name_ext)=} not in [str]!"
            )
            if name_ext[0] != "_":
                name_ext = f"_{name_ext}"
            new_property_name = f"{self.property_name}{name_ext}"

        if self.__class__.__name__ == "PDF":
            PDF_obj = globals()[self.__class__.__name__].from_1D_arr(
                new_property_name, new_input_arr, kwargs=new_kwargs
            )
        elif self.__class__.__name__ == "SED_fit_PDF":
            PDF_obj = globals()[self.__class__.__name__].from_1D_arr(
                new_property_name,
                new_input_arr,
                self.SED_fit_params,
                kwargs=new_kwargs,
            )
        elif self.__class__.__name__ == "Redshift_PDF":
            PDF_obj = globals()[self.__class__.__name__].from_1D_arr(
                new_input_arr, self.SED_fit_params, kwargs=new_kwargs
            )
        else:
            galfind_logger.critical(
                f"{self.__class__.__name__=} not in [PDF, SED_fit_PDF, Redshift_PDF]!"
            )
            breakpoint()
        # if chosen to save and it has a different name, save the PDF
        if (
            save
            and hasattr(self, "save_path")
            and new_property_name != self.property_name
        ):
            PDF_obj.save(
                self.save_path.replace(self.property_name, new_property_name)
            )
        return PDF_obj

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    # @classmethod
    # def from_ecsv(cls, path):
    #     try:
    #         tab = Table.read(path)
    #         property_name = tab.colnames[0]
    #         arr = np.array(tab[tab.colnames[0]]) * tab.meta["units"]
    #         kwargs = tab.meta
    #         for key in ["units", "size", "median", "l1_err", "u1_err"]:
    #             kwargs.pop(key)
    #         PDF_obj = cls.from_1D_arr(property_name, arr, kwargs)
    #         PDF_obj.save_path = path
    #         return PDF_obj
    #     except FileNotFoundError:
    #         return None
        
    @classmethod
    def from_npy(cls, path: str):
        """Load a previously saved `PDF` from disk.

        Parameters
        ----------
        path : `str`
            Path to the ``.npy`` file containing the underlying sample
            array, as previously written by `save`. The companion
            metadata file is expected at the same path with ``.npy``
            replaced by ``.meta.npy``.

        Returns
        -------
        `PDF`
            Reconstructed PDF, with `save_path` set to `path`.
        """
        arr = np.load(path)
        meta = np.load(path.replace(".npy", ".meta.npy"), allow_pickle=True).item()
        property_name = meta["name"]
        units = meta["units"]
        [meta.pop(name) for name in ["name", "units"]]
        PDF_obj = cls.from_1D_arr(property_name, arr * units, meta)
        PDF_obj.save_path = path
        return PDF_obj

    @classmethod
    def from_1D_arr(
        cls,
        property_name: str,
        arr: Union[u.Quantity, u.Magnitude, u.Dex],
        kwargs: dict = {},
        Nbins: int = 50,
        normed: bool = False,
        ignore_nans: bool = True,
    ):
        """Construct a `PDF` from a 1D array of sampled property values.

        Bins `arr` into a histogram of `Nbins` bins to build the
        `x`/`p_x` grid, and stores `arr` as `input_arr` on the
        returned object.

        Parameters
        ----------
        property_name : `str`
            Name of the galaxy property the array represents.
        arr : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            1D array of sampled values for the property.
        kwargs : `dict`, optional
            Additional metadata to store with the PDF. Default is `{}`.
        Nbins : `int`, optional
            Number of histogram bins used to construct the PDF.
            Default is `50`.
        normed : `bool`, optional
            Passed through to the `PDF` constructor; whether the
            resulting histogram density should be treated as already
            normalised. Default is `False`.
        ignore_nans : `bool`, optional
            Whether to discard non-finite values from `arr` before
            histogramming. Default is `True`.

        Returns
        -------
        `PDF`
            PDF constructed from the histogram of `arr`.

        Raises
        ------
        AssertionError
            If `arr` is not a `astropy.units.Quantity`,
            `astropy.units.Magnitude`, or `astropy.units.Dex`, or if
            no finite values remain after NaN-filtering.
        """
        assert isinstance(arr, (u.Quantity, u.Magnitude, u.Dex)), \
            galfind_logger.critical(
                f"{property_name=} 1D {arr=} with {type(arr)=}" + \
                " not in [u.Quantity, u.Magnitude, u.Dex]"
            )
        if ignore_nans:
            arr_ = arr[np.isfinite(arr)]
        else:
            arr_ = arr
        assert len(arr_) > 0, galfind_logger.critical(
            f"{property_name=} 1D {arr_=} with {len(arr_)=} == 0"
        )
        try:
            p_x, x_bin_edges = np.histogram(arr_.value.astype(np.float64), bins=Nbins, density=True)
        except:
            breakpoint()
        x = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1]) * arr_.unit
        PDF_obj = cls(property_name, x, p_x, kwargs, normed)
        if len(arr.shape) != 1:
            arr = arr.flatten()
        PDF_obj.input_arr = arr
        return PDF_obj

    @property
    def median(self):
        """`astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`: Median of the PDF.

        Computed and cached from `input_arr` if available (via
        `numpy.nanmedian`), otherwise derived from
        `get_percentile(50.0)`.
        """
        try:
            return self._median
        except AttributeError:
            if hasattr(self, "input_arr"):
                self._median = (
                    np.nanmedian(self.input_arr.value) * self.input_arr.unit
                )
            else:
                self._median = self.get_percentile(50.0)
            return self._median

    @property
    def errs(self):
        """`astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`: Asymmetric 1-sigma errors on `median`, as ``[lower, upper]``.

        Computed and cached from the 16th/84th percentiles of
        `input_arr` if available, otherwise from `get_percentile`.
        """
        try:
            return self._errs
        except AttributeError:
            if hasattr(self, "input_arr"):
                self._errs = [
                    self.median.value
                    - np.nanpercentile(self.input_arr.value, 16.0),
                    np.nanpercentile(self.input_arr.value, 84.0)
                    - self.median.value,
                ] * self.input_arr.unit
            else:
                self._errs = [
                    self.median.value - self.get_percentile(16.0).value,
                    self.get_percentile(84.0).value - self.median.value,
                ] * self.x.unit
            return self._errs

    def draw_sample(self, size: int = 10_000):
        """Draw a random sample of values from the PDF.

        Parameters
        ----------
        size : `int`, optional
            Number of samples to draw. Default is `10_000`.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            Random sample of `size` values drawn from `x`, weighted by
            the (normalised) probability `p_x`.
        """
        # draw a sample of specified size from the PDF
        return np.random.choice(self.x, size=size, p=self.p_x/np.sum(self.p_x)) * self.x.unit

    def integrate_between_lims(
        self,
        lower_x_lim: Union[int, float],
        upper_x_lim: Union[int, float],
    ):
        """Integrate the PDF between two limits using the trapezium rule.

        Parameters
        ----------
        lower_x_lim : `int` or `float`
            Lower integration limit, compared directly against
            `self.x` to find the nearest grid point.
        upper_x_lim : `int` or `float`
            Upper integration limit, compared directly against
            `self.x` to find the nearest grid point.

        Returns
        -------
        `float`
            Integral of `p_x` over `x` between the grid points nearest
            to `lower_x_lim` and `upper_x_lim`.
        """
        # find index of closest values in self.x to lower_x_lim and upper_x_lim
        index_x_min = np.argmin(np.absolute(self.x - lower_x_lim))
        index_x_max = np.argmin(np.absolute(self.x - upper_x_lim))
        # clip x/p_x distribution to integration limits
        x = self.x[index_x_min:index_x_max]
        p_x = self.p_x[index_x_min:index_x_max]
        # integrate using trapezium rule between limits
        return np.trapz(p_x, x)

    def get_peak(
        self: Self,
        nth_peak: int,
        log: bool = False,
    ) -> float:
        """Get the `nth_peak`-th peak of the PDF.

        Not fully implemented: peaks are only populated externally
        (e.g. via `SED_fit_PDF.load_peaks_from_best_fit`); if
        `self.peaks` has not yet been set, this initialises it and, if
        `nth_peak` is `0`, populates it with a placeholder peak whose
        ``'value'`` and ``'chi_sq'`` are both `None`.

        Parameters
        ----------
        nth_peak : `int`
            Index of the peak to retrieve.
        log : `bool`, optional
            If `True`, return a copy of the peak with its ``'value'``
            converted to ``log10`` (in dex). Default is `False`.

        Returns
        -------
        `dict`
            Dictionary with keys ``'value'`` and ``'chi_sq'`` for the
            requested peak.
        """
        # not properly implemented yet
        try:
            peaks = self.peaks[nth_peak]
        except (AttributeError, IndexError) as e:
            if isinstance(e, AttributeError):
                self.peaks = []
            # calculate the nth_peak - what if array isnt the correct length
            if nth_peak == 0:
                self.peaks.append({"value": None, "chi_sq": None})
        if log:
            peaks = deepcopy(self.peaks)
            peaks[nth_peak]["value"] = np.log10(peaks[nth_peak]["value"].value) * u.dex
        return peaks

        # currently just copied straight from Tom's plotting script
        # # calculate peak locations etc - should go inside of PDF class
        # pz_column, integral, peak_z, peak_loc, peak_second_loc, secondary_peak, ratio = useful_funcs_updated_new_galfind.robust_pdf([gal_id], [zbest], SED_code, field_name, rel_limits=True, z_fact=int_limit, use_custom_lephare_seds=custom_lephare, template=template, plot=False, version=catalog_version, custom_sex=custom_sex, min_percentage_err=min_percentage_err, custom_path=eazy_pdf_path, use_galfind=True)
        # print(integral, 'integral', peak_z, 'peak_z', peak_loc, 'peak_loc', peak_second_loc, 'peak_second_loc', secondary_peak, 'secondary_peak', ratio, 'ratio')

    def get_percentile(
        self: Self,
        percentile: float,
        log: bool = False
    ) -> float:
        """Get a given percentile of the PDF, with caching.

        Computed and cached from `input_arr` via
        `numpy.nanpercentile` if available, otherwise derived from the
        cumulative distribution of `p_x` over `x`.

        Parameters
        ----------
        percentile : `float`
            Percentile to compute, in the range [0, 100].
        log : `bool`, optional
            If `True`, return ``log10`` of the percentile value
            instead of the value itself. Default is `False`.

        Returns
        -------
        `float` or `astropy.units.Quantity`
            The requested percentile of the PDF, in the units of `x`
            (or a dimensionless `float` if `log` is `True`).

        Raises
        ------
        AssertionError
            If `percentile` is not a `float`.
        """
        assert isinstance(percentile, float), \
            galfind_logger.critical(
                f"{percentile=} with {type(percentile)=} != float"
            )
        try:
            perc = self.percentiles[f"{percentile:.1f}"]
        except (AttributeError, KeyError) as e:
            if isinstance(e, AttributeError):
                self.percentiles = {}
            if hasattr(self, "input_arr"):
                self.percentiles[f"{percentile:.1f}"] = np.nanpercentile( \
                    self.input_arr.value, percentile) * self.input_arr.unit
            else:
                # calculate percentile
                cdf = np.cumsum(self.p_x)
                cdf /= np.max(cdf)
                self.percentiles[f"{percentile:.1f}"] = (
                    float(
                        self.x.value[np.argmin(np.abs(cdf - percentile / 100.0))]
                    )
                    * self.x.unit
                )
            perc = self.percentiles[f"{percentile:.1f}"]
        if log:
            return np.log10(perc.value)
        else:
            return perc

    def manipulate_PDF(
        self,
        new_property_name: str,
        update_func: Callable[..., Union[list, np.array]],
        PDF_kwargs: dict = {},
        size: int = 10_000,
        **kwargs,
    ):
        """Create a new `PDF` by applying a transformation function to a sample.

        Draws (or takes the last `size` elements of `input_arr`) a
        sample of values from this PDF, applies `update_func` to it,
        and rebuilds a new PDF of the same class from the transformed
        sample.

        Parameters
        ----------
        new_property_name : `str`
            Property name for the new PDF.
        update_func : `Callable`
            Function applied to the sample array (plus any
            `**kwargs`) to produce the transformed sample.
        PDF_kwargs : `dict`, optional
            Additional metadata merged with `self.kwargs` for the new
            PDF. Default is `{}`.
        size : `int`, optional
            Size of the sample drawn/taken to transform. Default is
            `10_000`.
        **kwargs : `dict`
            Additional keyword arguments forwarded to `update_func`.

        Returns
        -------
        `PDF`
            New PDF constructed from the transformed sample.

        Raises
        ------
        AssertionError
            If the drawn/taken sample does not have length `size`.
        """
        if hasattr(self, "input_arr"):
            # take the last 'size' elements of the input array
            sample = self.input_arr[-size:]
        else:
            sample = self.draw_sample(size)
        assert (
            len(sample) == size
        )  # ensures size > len(sample) throws an error
        updated_sample = update_func(
            sample, **kwargs
        )  # [update_func(val, **kwargs) for val in sample]
        return self.__class__.from_1D_arr(
            new_property_name, updated_sample, {**self.kwargs, **PDF_kwargs}
        )

    def save(
        self: Self, 
        save_path: str, 
        size: int = 10_000,
    ) -> None:
        """Save the PDF's underlying sample and metadata to disk.

        Writes the sample array (`input_arr` if available, otherwise a
        freshly drawn sample of size `size`) to ``<save_path>.npy``,
        and its metadata (`kwargs` plus property name and units) to
        ``<save_path>.meta.npy``.

        Parameters
        ----------
        save_path : `str`
            Path to save the PDF to. A ``.npy`` extension is appended
            if not already present.
        size : `int`, optional
            Number of samples to draw if `input_arr` is not already
            set. Default is `10_000`.
        """
        if hasattr(self, "input_arr"):
            save_arr = self.input_arr
        else:
            save_arr = self.draw_sample(size)
        meta = {
            **self.kwargs,
            **{
                "name": self.property_name,
                "units": self.x.unit,
            },
        }
        save_arr = np.array(save_arr.value)
        if save_path[-4:] != ".npy":
            save_path += ".npy"
        self.save_path = save_path
        funcs.make_dirs(save_path)
        np.save(save_path, save_arr)
        meta_path = save_path.replace(".npy", ".meta.npy")
        np.save(meta_path, meta)
        funcs.change_file_permissions(save_path)
        funcs.change_file_permissions(meta_path)

    def add_save_path(self, path):  # -> self
        """Set `save_path` on this PDF without writing to disk.

        Parameters
        ----------
        path : `str`
            Path to associate with this PDF as its `save_path`.

        Returns
        -------
        `PDF`
            This PDF instance, with `save_path` set.
        """
        self.save_path = path
        return self

    def plot(
        self,
        ax,
        annotate: bool = True,
        #annotate_peak_loc: bool = False,
        colour: str = "black",
        log: bool = False,
        hatch: str = "//",
        label_kwargs: Dict[str, Any] = {},
        **pdf_kwargs,
    ) -> None:
        """Plot the PDF as a filled curve on a matplotlib axis.

        Interpolates the PDF onto a grid derived from a drawn/stored
        sample, plots it as a line with a shaded region beneath, sets
        the axis limits and x-axis label, and (if `annotate` is
        `True`) marks the peak location and 1-sigma percentile range
        with vertical lines and text annotations, including the
        property value and chi-squared of the best-fit peak.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axis to plot the PDF onto.
        annotate : `bool`, optional
            Whether to annotate the plot with the peak value, 1-sigma
            percentile lines, and chi-squared. Default is `True`.
        colour : `str`, optional
            Colour used for the line, shaded region, and annotations.
            Default is `'black'`.
        log : `bool`, optional
            Whether to plot the PDF in ``log10`` of the property
            value. Default is `False`.
        hatch : `str`, optional
            Matplotlib hatch pattern used for the shaded region
            beneath the curve. Default is `'//'`.
        label_kwargs : `dict`, optional
            Additional keyword arguments passed to `ax.set_xlabel`.
            Default is `{}`.
        **pdf_kwargs : `dict`
            Additional keyword arguments passed to `ax.plot` for the
            PDF curve.

        Returns
        -------
        None
        """

        if not hasattr(self, "input_arr"):
            input_arr = self.draw_sample(10_000)
        else:
            input_arr = self.input_arr

        if isinstance(input_arr, tuple([u.Quantity, u.Magnitude, u.Dex])):
            input_arr = input_arr.value
        if log:
            input_arr = np.log10(input_arr)
        
        # construct gaussian_kde
        # kde = gaussian_kde(input_arr)
        x = np.linspace(
            np.min(input_arr),
            np.max(input_arr),
            len(input_arr)
        )
        # y = kde(x)
        #x = np.sort(input_arr)
        y = np.interp(x, self.x.value, self.p_x)

        # plot the pdf
        ax.plot(
            x,
            y,
            color = colour,
            **pdf_kwargs
        )
        ax.fill_between(
            x,
            y,
            color = colour,
            alpha = 0.2,
            hatch = hatch
        )

        perc = {}
        perc_lims = [1.0, 3.0, 97.0, 99.0]
        if annotate:
            perc_lims += [16.0, 84.0]
        for p in perc_lims:
            perc_ = self.get_percentile(p, log = log)
            if isinstance(perc_, u.Quantity):
                perc_ = perc_.value
            perc[p] = perc_

        # Set x and y plot limits
        ax.set_xlim(np.max([np.min(x), perc[1.0]]), np.max(x)) #np.min([np.max(x), perc[99.0]]))
        ax.set_ylim(0, 1.1 * np.max(y))

        if self.property_name in funcs.property_name_to_label:
            x_plot_name = funcs.property_name_to_label[self.property_name]
        else:
            x_plot_name = self.property_name
            galfind_logger.warning(
                f"{self.property_name=} not in funcs.property_name_to_label, using property name as label."
            )
        if log:
            x_plot_name = r"$\log_{10}($" + x_plot_name + r"$)$"
        label_kwargs.setdefault("fontsize", "medium")
        ax.set_xlabel(x_plot_name, **label_kwargs)
        # turn off grid
        ax.grid(False)
        # turn off y axis tick labels
        ax.tick_params(axis="y", which="both", labelleft=False, labelright=False)

        if annotate:
            # Draw vertical line at zbest
            ax.axvline(
                self.get_peak(0, log = log)["value"],
                color=colour,
                linestyle="--",
                alpha=0.5,
                lw=2,
            )
            ax.axvline(
                perc[16.0],
                color=colour,
                linestyle=":",
                alpha=0.5,
                lw=2,
            )
            ax.axvline(
                perc[84.0],
                color=colour,
                linestyle=":",
                alpha=0.5,
                lw=2,
            )
            ax.annotate(
                r"-1$\sigma$",
                (perc[16.0], 0.1),
                fontsize="small",
                ha="center",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
            ax.annotate(
                r"+1$\sigma$",
                (perc[84.0], 0.1),
                fontsize="small",
                ha="center",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
            ax.text(
                0.05,
                0.95,
                r"$z_{\rm phot}="
                + f'{self.get_peak(0, log = log)["value"].value:.1f}'
                + f'^{{+{(perc[84.0] - self.get_peak(0, log = log)["value"].value):.1f}}}_{{-{(self.get_peak(0, log = log)["value"].value - perc[16.]):.1f}}}$',
                transform=ax.transAxes,
                #(self.get_peak(0, log = log)["value"], 1.17),
                fontsize="medium",
                va="top",
                ha="left",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

            # Horizontal arrow at PDF peak going left or right depending on which side PDF is on, labelled with chi2
            # Check if highest peak is closer to xlim[0] or xlim[1]
            #x_lim = ax.get_xlim()
            #y_lim = ax.get_ylim()
            # amount = 0.3 * (x_lim[1] - x_lim[0])
            # if (
            #     self.get_peak(0, log = log)["value"] - x_lim[0]
            #     < x_lim[1] - self.get_peak(0, log = log)["value"]
            # ):
            #     direction = 1
            # else:
            #     direction = -1
            ax.text(
                0.05,
                0.80,
                r"$\chi^2=$" + f'{self.get_peak(0, log = log)["chi_sq"]:.2f}',
                #(self.get_peak(0, log = log)["value"], 1.0),
                transform=ax.transAxes,
                #xytext=(self.get_peak(0, log = log)["value"] + direction * amount, 0.90),
                fontsize="small",
                va="top",
                ha="left",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                # arrowprops=dict(
                #     facecolor=colour,
                #     edgecolor=colour,
                #     arrowstyle="-|>",
                #     lw=1.5,
                #     path_effects=[
                #         pe.withStroke(linewidth=1, foreground="white")
                #     ],
                # ),
            )

            # annotate PDF with peak locations etc
            # if annotate_peak_loc:
            #     ax.scatter(self.get_peak(0)["value"], peak_pdf, color = colour, edgecolors = colour, marker='o', facecolor='none')

            #     secondary_peak = self.get_peak(1)["value"]
            #     if secondary_peak > 0:
            #         ax.scatter(secondary_peak, secondary_peak_pdf, edgecolor='orange', marker='o', facecolor='none')
            #         ax.annotate(f'P(S)/P(P): {ratio:.2f}', loc_ratio, fontsize='x-small')

            # ax.annotate(f'$\\sum = {float(integral):.2f}$', (zbest, 0.45), fontsize='small', \
            # transform = ax.get_yaxis_transform(), va='bottom', ha='center', fontweight='bold', \
            # color=eazy_color, path_effects=[pe.withStroke(linewidth=3, foreground='white')])


class SED_fit_PDF(PDF):
    """`PDF` subclass associated with a specific SED fitting code run.

    Extends `PDF` by tracking the `SED_fit_params` of the SED fitting
    run that produced this PDF, and by providing helpers to populate
    peak locations from `SED_result` best-fit values.

    Parameters
    ----------
    property_name : `str`
        Name of the galaxy property this PDF represents.
    x : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
        Grid of property values at which `p_x` is defined.
    p_x : `numpy.ndarray`
        Probability density evaluated at each point in `x`.
    SED_fit_params : `dict`
        SED fitting parameters/options of the run that produced this
        PDF.
    kwargs : `dict`, optional
        Additional metadata to store with the PDF. Default is `{}`.
    normed : `bool`, optional
        Whether `p_x` is already normalised. Default is `False`.

    Attributes
    ----------
    SED_fit_params : `dict`
        SED fitting parameters/options of the run that produced this
        PDF.
    """

    def __init__(
        self,
        property_name,
        x,
        p_x,
        SED_fit_params,
        kwargs={},
        normed=False,
    ):
        self.SED_fit_params = SED_fit_params
        super().__init__(property_name, x, p_x, kwargs, normed)

    @classmethod
    def from_1D_arr(
        cls,
        property_name,
        arr,
        SED_fit_params,
        kwargs={},
        Nbins=50,
        normed=False,
    ):
        """Construct a `SED_fit_PDF` from a 1D array of sampled property values.

        Parameters
        ----------
        property_name : `str`
            Name of the galaxy property the array represents.
        arr : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            1D array of sampled values for the property.
        SED_fit_params : `dict`
            SED fitting parameters/options of the run that produced
            `arr`.
        kwargs : `dict`, optional
            Additional metadata to store with the PDF. Default is `{}`.
        Nbins : `int`, optional
            Number of histogram bins used to construct the PDF.
            Default is `50`.
        normed : `bool`, optional
            Whether `arr`'s histogram density should be treated as
            already normalised. Default is `False`.

        Returns
        -------
        `SED_fit_PDF`
            PDF constructed from the histogram of `arr`, with
            `input_arr` set to `arr`.
        """
        # super doesn't work here due to argument differences between PDF().__init__ and SED_fit_PDF().__init__
        PDF_obj = PDF.from_1D_arr(
            property_name, arr, kwargs, Nbins, normed
        )  # normalizes here if not already
        sed_fit_PDF = cls(
            property_name,
            PDF_obj.x,
            PDF_obj.p_x,
            SED_fit_params,
            kwargs,
            True,
        )
        sed_fit_PDF.input_arr = arr
        return sed_fit_PDF

    def load_peaks_from_SED_result(self, SED_result, nth_peak=0):
        """Populate the 0th peak of this PDF from an `SED_result`'s best-fit values.

        Parameters
        ----------
        SED_result : `SED_result`
            SED fitting result providing the best-fit property value
            (via ``SED_result.properties[self.property_name]``) and
            chi-squared (via ``SED_result.properties['chi_sq']``).
        nth_peak : `int`, optional
            Index of the peak to populate; must currently be `0`.
            Default is `0`.

        Returns
        -------
        `SED_fit_PDF`
            This PDF instance, with its 0th peak populated (via
            `load_peaks_from_best_fit`).

        Raises
        ------
        AssertionError
            If `nth_peak` is not an `int`, or is not `0`.
        """
        assert isinstance(nth_peak, int), galfind_logger.critical(
            f"nth_peak with type = {type(nth_peak)} must be of type 'int'"
        )
        assert nth_peak == 0, galfind_logger.critical(
            f"SED_fit_PDF.load_peaks_from_SED_result only loads the 0th peak, not the {funcs.ordinal(nth_peak)}"
        )
        # TODO: Implement _dicts_equal from funcs
        # if isinstance(self.SED_fit_params, dict):
        #     # ensure all keys in self.SED_fit_params are in SED_result.SED_code.SED_fit_params and vice versa
        #     assert all(
        #         key in SED_result.SED_code.SED_fit_params for key in self.SED_fit_params.keys()
        #     ), galfind_logger.critical(
        #         f"{self.SED_fit_params.keys()} not all in {SED_result.SED_code.SED_fit_params.keys()}"
        #     )
        #     assert all(
        #         key in self.SED_fit_params for key in SED_result.SED_code.SED_fit_params.keys()
        #     ), galfind_logger.critical(
        #         f"{SED_result.SED_code.SED_fit_params.keys()} not all in {self.SED_fit_params.keys()}"
        #     )
        #     # ensure all values for each key are the same
        #     for key in self.SED_fit_params.keys():
        #         assert (
        #             SED_result.SED_code.SED_fit_params[key] == self.SED_fit_params[key]
        #         ), galfind_logger.critical(
        #             f"{SED_result.SED_code.SED_fit_params[key]=} != {self.SED_fit_params[key]=}"
        #         )
        # else:
        #     assert (
        #         SED_result.SED_code.SED_fit_params == self.SED_fit_params
        #     ), galfind_logger.critical(
        #         f"{SED_result.SED_code.SED_fit_params=} != {self.SED_fit_params=}"
        #     )
        # load peak value and peak chi_sq
        self.load_peaks_from_best_fit(
            SED_result.properties[self.property_name],
            SED_result.properties["chi_sq"],
        )
        return self

    def load_peaks_from_best_fit(self, property, chi_sq):
        """Set the 0th peak of this PDF from an explicit value and chi-squared.

        Parameters
        ----------
        property : `Any`
            Best-fit property value to store as the 0th peak's value.
        chi_sq : `float`
            Chi-squared of the best fit, stored alongside the peak
            value.

        Returns
        -------
        `SED_fit_PDF`
            This PDF instance, with its 0th peak set.
        """
        zeroth_peak = {"value": property, "chi_sq": chi_sq}
        if not hasattr(self, "peaks"):
            self.peaks = []
        if len(self.peaks) > 0:
            self.peaks[0] = zeroth_peak
        else:
            self.peaks.append(zeroth_peak)
        return self


class Redshift_PDF(SED_fit_PDF):
    """`SED_fit_PDF` subclass specifically for redshift posterior distributions.

    Fixes `property_name` to ``'z'``.

    Parameters
    ----------
    z : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
        Grid of redshift values at which `p_z` is defined.
    p_z : `numpy.ndarray`
        Probability density evaluated at each point in `z`.
    SED_fit_params : `dict`
        SED fitting parameters/options of the run that produced this
        PDF.
    kwargs : `dict`, optional
        Additional metadata to store with the PDF. Default is `{}`.
    normed : `bool`, optional
        Whether `p_z` is already normalised. Default is `False`.
    """

    def __init__(
        self,
        z,
        p_z,
        SED_fit_params,
        kwargs={},
        normed=False
    ):
        super().__init__("z", z, p_z, SED_fit_params, kwargs, normed)

    @classmethod
    def from_1D_arr(
        cls,
        z_arr,
        SED_fit_params,
        kwargs={},
        Nbins=50,
        normed=False,
    ):
        """Construct a `Redshift_PDF` from a 1D array of sampled redshift values.

        Parameters
        ----------
        z_arr : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            1D array of sampled redshift values.
        SED_fit_params : `dict`
            SED fitting parameters/options of the run that produced
            `z_arr`.
        kwargs : `dict`, optional
            Additional metadata to store with the PDF. Default is `{}`.
        Nbins : `int`, optional
            Number of histogram bins used to construct the PDF.
            Default is `50`.
        normed : `bool`, optional
            Whether `z_arr`'s histogram density should be treated as
            already normalised. Default is `False`.

        Returns
        -------
        `Redshift_PDF`
            Redshift PDF constructed from the histogram of `z_arr`,
            with `input_arr` set to `z_arr`.
        """
        SED_fit_PDF_obj = SED_fit_PDF.from_1D_arr(
            "z", z_arr, SED_fit_params, kwargs, Nbins, normed
        )  # normalized here if not already
        z_PDF = cls(
            SED_fit_PDF_obj.x,
            SED_fit_PDF_obj.p_x,
            SED_fit_params,
            kwargs,
            True,
        )
        z_PDF.input_arr = z_arr
        return z_PDF

    # @classmethod
    # def from_SED_code_output(cls, data_path, ID, code, SED_fit_params):
    #     z, p_z = code.extract_z_PDF(data_path, ID)
    #     return cls(z, p_z, SED_fit_params)

    def integrate_between_lims(
        self,
        delta_z_over_z: float,
        zbest: Optional[float] = None,
        z_min: float = 0.,
        z_max: float = 25.,
    ):
        """Integrate the redshift PDF within a fractional window around the best-fit redshift.

        Parameters
        ----------
        delta_z_over_z : `float`
            Fractional redshift window; integration limits are
            ``zbest * (1 ± delta_z_over_z)``.
        zbest : `float`, optional
            Best-fit redshift to center the window on. If `None`, the
            0th peak of the PDF (via `get_peak`) is used. Default is
            `None`.
        z_min : `float`, optional
            Minimum allowed redshift; integration limits are clipped
            to this value. Default is `0.`.
        z_max : `float`, optional
            Maximum allowed redshift; integration limits are clipped
            to this value. Default is `25.`.

        Returns
        -------
        `float`
            Integral of the PDF (via `PDF.integrate_between_lims`)
            between the clipped lower and upper redshift limits.
        """
        # find best fitting redshift from peak of the PDF distribution - not needed if peak is loaded in PDF object
        if zbest is None:
            zbest = self.get_peak(0)["value"]  # find first peak
        elif isinstance(zbest, (int, float)):  # correct format
            pass
        else:
            galfind_logger.critical(
                f"zbest = {zbest} with type = {type(zbest)} is not in [int, float, None]!"
            )
        # calculate redshift limits
        lower_z_lim = np.clip(zbest * (1 - delta_z_over_z), z_min, z_max)
        upper_z_lim = np.clip(zbest * (1 + delta_z_over_z), z_min, z_max)
        return super().integrate_between_lims(lower_z_lim, upper_z_lim)


class PDF_nD:
    """Joint N-dimensional representation of multiple 1D `PDF` objects.

    Bundles several `PDF` instances (assumed to share the same number
    of drawn/stored samples) so that a function of several galaxy
    properties can be evaluated over their joint (unnormalised) sample
    chains.

    Parameters
    ----------
    ordered_PDFs : `list` of `PDF`
        Ordered sequence of `PDF` objects to combine, each of which
        must already have an `input_arr` attribute of the same length.

    Attributes
    ----------
    dimensions : `int`
        Number of `PDF` objects combined (i.e. the dimensionality).
    PDFs : `list` of `PDF`
        The combined `PDF` objects, in the order given.
    """

    def __init__(self, ordered_PDFs):
        # ensure all PDFs have input arr of values, all of which are the same length
        try:
            assert all(
                hasattr(PDF_obj, "input_arr") for PDF_obj in ordered_PDFs
            )
        except:
            breakpoint()
        assert all(
            len(PDF_obj.input_arr) == len(ordered_PDFs[0].input_arr)
            for PDF_obj in ordered_PDFs
        )
        self.dimensions = len(ordered_PDFs)
        self.PDFs = ordered_PDFs

    @classmethod
    def from_matrix(cls, property_names, matrix):
        """Construct a `PDF_nD` from a matrix of per-property sample rows.

        Parameters
        ----------
        property_names : `list` of `str`
            Names of the galaxy properties, one per row of `matrix`.
        matrix : `numpy.ndarray`
            2D array of shape ``(n_properties, n_samples)``; each row
            is histogrammed into a `PDF` via `PDF.from_1D_arr`.

        Returns
        -------
        `PDF_nD`
            Joint PDF combining one `PDF` per row of `matrix`.

        Raises
        ------
        AssertionError
            If `len(property_names)` does not match `matrix.shape[0]`.
        """
        assert len(property_names) == matrix.shape[0]  # 0 or 1 here, not sure
        ordered_PDFs = [
            PDF.from_1D_arr(property_name, row)
            for property_name, row in zip(property_names, matrix)
        ]
        return cls(ordered_PDFs)

    def __len__(self):
        return len(self.PDFs[0])

    def __call__(self, func, independent_var, size=None, output_type="chains"):
        # need to provide additional assertions here too
        # assert that the dimensions of PDF_nD must be the same as the input arguments - 1 of func
        chains = np.array(
            [
                func(independent_var, *vals)
                for vals in np.array(
                    [PDF_obj.input_arr for PDF_obj in self.PDFs]
                ).T
            ]
        )
        assert chains.shape == (len(self), len(independent_var))
        if size is None:
            pass
        elif isinstance(size, int):
            chains = chains[-size:]
        else:
            galfind_logger.critical(
                f"{type(size)=} not in [None, int, np.int]!"
            )
        assert output_type in ["chains", "percentiles"]
        if output_type == "chains":
            return chains
        elif output_type == "percentiles":
            func_l1_med_u1 = [
                np.percentile(chains[:, i], [16.0, 50.0, 84.0])
                for i in range(len(independent_var))
            ]
            return [
                func_l1_med_u1[:, 0],
                func_l1_med_u1[:, 1],
                func_l1_med_u1[:, 2],
            ]

    def plot_corner(self):
        pass

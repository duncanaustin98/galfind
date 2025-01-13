from __future__ import annotations

import time
from copy import deepcopy
import astropy.units as u
import matplotlib.patheffects as pe
import numpy as np
from astropy.table import Table
from typing import Callable, Union, Optional, TYPE_CHECKING
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import config, galfind_logger
from . import useful_funcs_austind as funcs


class PDF:
    def __init__(
        self,
        property_name,
        x,
        p_x,
        kwargs={},
        normed: bool = False,
        timed: bool = False,
    ):
        if timed:
            start = time.time()
        if type(x) not in [u.Quantity, u.Magnitude, u.Dex]:
            breakpoint()
        # assert type(x) in [u.Quantity, u.Magnitude, u.Dex]
        self.property_name = property_name
        self.x = x
        self.kwargs = kwargs
        if timed:
            mid = time.time()
        # normalize to np.trapz(p_x, x) == 1
        if not normed:
            p_x /= np.trapz(p_x, x.value)
        self.p_x = p_x
        if timed:
            end = time.time()
            # print(mid - start, end - mid)

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
        other: Union["PDF", int, float, u.Quantity, u.Magnitude, u.Dex],
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
        if type(other) in [int, float, u.Quantity, u.Magnitude, u.Dex]:
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

        if type(name_ext) == type(None):
            new_property_name = self.property_name
        else:  # type(name_ext) == str
            assert type(name_ext) in [str], galfind_logger.critical(
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
        timed: bool = False,
    ):
        assert isinstance(arr, (u.Quantity, u.Magnitude, u.Dex)), \
            galfind_logger.critical(
                f"{property_name=} 1D {arr=} with {type(arr)=}" + \
                " not in [u.Quantity, u.Magnitude, u.Dex]"
            )
        if ignore_nans:
            arr_ = arr[~np.isnan(arr)]
        else:
            arr_ = arr
        assert len(arr_) > 0, galfind_logger.critical(
            f"{property_name=} 1D {arr_=} with {len(arr_)=} == 0"
        )
        try:
            p_x, x_bin_edges = np.histogram(arr_.value, bins=Nbins, density=True)
        except:
            breakpoint()
        x = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1]) * arr_.unit
        PDF_obj = cls(property_name, x, p_x, kwargs, normed, timed)
        PDF_obj.input_arr = arr
        return PDF_obj

    @property
    def median(self):
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
        # draw a sample of specified size from the PDF
        return np.random.choice(self.x, size=size, p=self.p_x/np.sum(self.p_x)) * self.x.unit

    def integrate_between_lims(
        self, lower_x_lim: Union[int, float], upper_x_lim: Union[int, float]
    ):
        # find index of closest values in self.x to lower_x_lim and upper_x_lim
        index_x_min = np.argmin(np.absolute(self.x - lower_x_lim))
        index_x_max = np.argmin(np.absolute(self.x - upper_x_lim))
        # clip x/p_x distribution to integration limits
        x = self.x[index_x_min:index_x_max]
        p_x = self.p_x[index_x_min:index_x_max]
        # integrate using trapezium rule between limits
        return np.trapz(p_x, x)

    def get_peak(self, nth_peak: int):
        # not properly implemented yet
        try:
            self.peaks[nth_peak]
        except (AttributeError, IndexError) as e:
            if type(e) == AttributeError:
                self.peaks = []
            # calculate the nth_peak - what if array isnt the correct length
            if nth_peak == 0:
                self.peaks.append({"value": None, "chi_sq": None})
        return self.peaks[nth_peak]

        # currently just copied straight from Tom's plotting script
        # # calculate peak locations etc - should go inside of PDF class
        # pz_column, integral, peak_z, peak_loc, peak_second_loc, secondary_peak, ratio = useful_funcs_updated_new_galfind.robust_pdf([gal_id], [zbest], SED_code, field_name, rel_limits=True, z_fact=int_limit, use_custom_lephare_seds=custom_lephare, template=template, plot=False, version=catalog_version, custom_sex=custom_sex, min_percentage_err=min_percentage_err, custom_path=eazy_pdf_path, use_galfind=True)
        # print(integral, 'integral', peak_z, 'peak_z', peak_loc, 'peak_loc', peak_second_loc, 'peak_second_loc', secondary_peak, 'secondary_peak', ratio, 'ratio')

    def get_percentile(self, percentile: float):
        assert isinstance(percentile, float), \
            galfind_logger.critical(
                f"{percentile=} with {type(percentile)=} != float"
            )
        try:
            return self.percentiles[f"{percentile:.1f}"]
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
            return self.percentiles[f"{percentile:.1f}"]

    def manipulate_PDF(
        self,
        new_property_name: str,
        update_func: Callable[..., Union[list, np.array]],
        PDF_kwargs: dict = {},
        size: int = 10_000,
        **kwargs,
    ):
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
        self.save_path = path
        return self

    def plot(
        self,
        ax,
        annotate: bool = True,
        annotate_peak_loc: bool = False,
        colour: str = "black",
    ) -> None:
        ax.plot(self.x, self.p_x / np.max(self.p_x), color=colour)

        # Set x and y plot limits
        ax.set_xlim(
            self.get_percentile(3.0) - 0.2, self.get_percentile(97.0) + 0.2
        )
        if abs(ax.get_xlim()[1] - ax.get_xlim()[0]) < 0.3:
            ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)
        ax.set_ylim(0, 1.2)

        # fill inside PDF with hatch
        x_lim = np.linspace(
            self.get_percentile(1.0), self.get_percentile(99.0)
        )  # np.linspace(0.93 * float(self.get_peak(0)["value"]), 1.07 * float(self.get_peak(0)["value"]), 100)

        pdf_lim = np.interp(x_lim, self.x, self.p_x / np.max(self.p_x))
        ax.fill_between(x_lim, pdf_lim, color=colour, alpha=0.2, hatch="//")

        ax.grid(False)

        if annotate:
            # Draw vertical line at zbest
            ax.axvline(
                self.get_peak(0)["value"],
                color=colour,
                linestyle="--",
                alpha=0.5,
                lw=2,
            )
            ax.axvline(
                self.get_percentile(16.0),
                color=colour,
                linestyle=":",
                alpha=0.5,
                lw=2,
            )
            ax.axvline(
                self.get_percentile(84.0),
                color=colour,
                linestyle=":",
                alpha=0.5,
                lw=2,
            )
            ax.annotate(
                r"-1$\sigma$",
                (self.get_percentile(16.0), 0.1),
                fontsize="small",
                ha="center",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
            ax.annotate(
                r"+1$\sigma$",
                (self.get_percentile(84.0), 0.1),
                fontsize="small",
                ha="center",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
            ax.annotate(
                r"$z_{\rm phot}=$"
                + f'{self.get_peak(0)["value"]:.1f}'
                + f'$^{{+{(self.get_percentile(84.) - self.get_peak(0)["value"]):.1f}}}_{{-{(self.get_peak(0)["value"] - self.get_percentile(16.)):.1f}}}$',
                (self.get_peak(0)["value"], 1.17),
                fontsize="medium",
                va="top",
                ha="center",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

            # Horizontal arrow at PDF peak going left or right depending on which side PDF is on, labelled with chi2
            # Check if highest peak is closer to xlim[0] or xlim[1]
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            amount = 0.3 * (x_lim[1] - x_lim[0])
            if (
                self.get_peak(0)["value"] - x_lim[0]
                < x_lim[1] - self.get_peak(0)["value"]
            ):
                direction = 1
            else:
                direction = -1
            ax.annotate(
                r"$\chi^2=$" + f'{self.get_peak(0)["chi_sq"]:.2f}',
                (self.get_peak(0)["value"], 1.0),
                xytext=(self.get_peak(0)["value"] + direction * amount, 0.90),
                fontsize="small",
                va="top",
                ha="center",
                color=colour,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                arrowprops=dict(
                    facecolor=colour,
                    edgecolor=colour,
                    arrowstyle="-|>",
                    lw=1.5,
                    path_effects=[
                        pe.withStroke(linewidth=1, foreground="white")
                    ],
                ),
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
    def __init__(
        self,
        property_name,
        x,
        p_x,
        SED_fit_params,
        kwargs={},
        normed=False,
        timed=False,
    ):
        self.SED_fit_params = SED_fit_params
        super().__init__(property_name, x, p_x, kwargs, normed, timed)

    @classmethod
    def from_1D_arr(
        cls,
        property_name,
        arr,
        SED_fit_params,
        kwargs={},
        Nbins=50,
        normed=False,
        timed=False,
    ):
        # super doesn't work here due to argument differences between PDF().__init__ and SED_fit_PDF().__init__
        PDF_obj = PDF.from_1D_arr(
            property_name, arr, kwargs, Nbins, normed, timed
        )  # normalizes here if not already
        sed_fit_PDF = cls(
            property_name,
            PDF_obj.x,
            PDF_obj.p_x,
            SED_fit_params,
            kwargs,
            True,
            timed,
        )
        sed_fit_PDF.input_arr = arr
        return sed_fit_PDF

    def load_peaks_from_SED_result(self, SED_result, nth_peak=0):
        assert type(nth_peak) == int, galfind_logger.critical(
            f"nth_peak with type = {type(nth_peak)} must be of type 'int'"
        )
        assert nth_peak == 0, galfind_logger.critical(
            f"SED_fit_PDF.load_peaks_from_SED_result only loads the 0th peak, not the {funcs.ordinal(nth_peak)}"
        )
        assert (
            SED_result.SED_fit_params == self.SED_fit_params
        ), galfind_logger.critical(
            f"SED_result.SED_fit_params = {SED_result.SED_fit_params} != self.SED_fit_params = {self.SED_fit_params}"
        )
        # load peak value and peak chi_sq
        self.load_peaks_from_best_fit(
            SED_result.properties[self.property_name],
            SED_result.properties["chi_sq"],
        )
        return self

    def load_peaks_from_best_fit(self, property, chi_sq):
        zeroth_peak = {"value": property, "chi_sq": chi_sq}
        if not hasattr(self, "peaks"):
            self.peaks = []
        if len(self.peaks) > 0:
            self.peaks[0] = zeroth_peak
        else:
            self.peaks.append(zeroth_peak)
        return self


class Redshift_PDF(SED_fit_PDF):
    def __init__(
        self, z, p_z, SED_fit_params, kwargs={}, normed=False, timed=False
    ):
        super().__init__("z", z, p_z, SED_fit_params, kwargs, normed, timed)

    @classmethod
    def from_1D_arr(
        cls,
        z_arr,
        SED_fit_params,
        kwargs={},
        Nbins=50,
        normed=False,
        timed=False,
    ):
        SED_fit_PDF_obj = SED_fit_PDF.from_1D_arr(
            "z", z_arr, SED_fit_params, kwargs, Nbins, normed, timed
        )  # normalized here if not already
        z_PDF = cls(
            SED_fit_PDF_obj.x,
            SED_fit_PDF_obj.p_x,
            SED_fit_params,
            kwargs,
            True,
            timed,
        )
        z_PDF.input_arr = z_arr
        return z_PDF

    # @classmethod
    # def from_SED_code_output(cls, data_path, ID, code, SED_fit_params):
    #     z, p_z = code.extract_z_PDF(data_path, ID)
    #     return cls(z, p_z, SED_fit_params)

    def integrate_between_lims(
        self,
        delta_z_over_z,
        zbest: Optional[float] = None,
        z_min: float = 0.,
        z_max: float = 25.,
    ):
        # find best fitting redshift from peak of the PDF distribution - not needed if peak is loaded in PDF object
        if type(zbest) == type(None):
            zbest = self.get_peak(0)["value"]  # find first peak
        elif type(zbest) in [int, float]:  # correct format
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

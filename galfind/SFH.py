import h5py
from copy import deepcopy
from pathlib import Path
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.pyplot as plt
import astropy.units as u
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Callable, Union
try:
    from typing import Self #, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self #, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger

class SFH:

    def __init__(
        self: Self,
        z: float,
        ages: NDArray[float],
        sfh_post: NDArray[float],
        type: str = "continuity_bursty",
    ):
        self.z = z
        self.ages = ages
        self.sfh_post = sfh_post
        self.type = type

    @classmethod
    def from_pipes_post(cls: Self, path: str) -> Self:
        
        if not Path(path).is_file():
            err_message = f"SFH file {path} not found!"
            galfind_logger.critical(err_message)
            raise FileNotFoundError(err_message)
        
        import ast
        import bagpipes as pipes
        with h5py.File(path, "r") as h5:
            basic_quantities = h5["basic_quantities"]
            assert "sfh" in dict(basic_quantities).keys(), \
                galfind_logger.critical(
                    f"'SFH' not found in {path} basic quantites! "
                )
            sfh = np.array(basic_quantities["sfh"])
            fit_instructions = ast.literal_eval(h5.attrs["fit_instructions"])
            # extract z
            if isinstance(fit_instructions["redshift"], float):
                z = fit_instructions["redshift"]
            else:
                # extract median redshift from the posterior
                advanced_quantities = h5["advanced_quantities"]
                assert "redshift" in dict(advanced_quantities).keys(), \
                    galfind_logger.critical(
                        f"'redshift' not found in {path} advanced quantites!"
                    )
                z = np.median(advanced_quantities["redshift"])
            h5.close()
        
        # extract SFH type
        remove_keys = ["t_bc", "dust", "nebular", "redshift"]
        sfh_type = [
            key for key in fit_instructions.keys() 
            if not any(remove_substr in key for remove_substr in remove_keys)
        ]
        assert len(sfh_type) == 1, \
            galfind_logger.critical(
                f"SFH type not uniquely defined in {path} fit instructions!"
            )
        sfh_type = sfh_type[0]

        # extract time array (DIRECTLY FROM BAGPIPES)
        log_sampling = 0.0025 # default
        hubble_time = pipes.utils.age_at_z[pipes.utils.z_array == 0.]
        # Set up the age sampling for internal SFH calculations.
        log_age_max = np.log10(hubble_time) + 9. + 2 * log_sampling
        ages = 10 ** np.arange(6., log_age_max, log_sampling) * u.yr

        return cls(z, ages, sfh, sfh_type)

    def __repr__(self: Self) -> str:
        return f"SFH(z={self.z:.2f}, type={self.type})"
    
    @property
    def age_of_universe(self: Self) -> u.Quantity:
        import bagpipes as pipes
        return 1.e9 * np.interp(self.z, pipes.utils.z_array, pipes.utils.age_at_z) * u.yr

    def plot(
        self: Self,
        ax: plt.Axes,
        time_units: u.Unit = u.Myr,
        plot_type: str = "lookback",
        z_axis: bool = True,
        label_z: bool = True,
        annotate: bool = True,
        save: bool = False,
        primary_colour: str = "black",
        secondary_colour: str = "gray",
        zvals: List[Union[float, int]] = [0, 0.5, 1, 2, 4, 6, 7, 8, 10, 16, 25],
        zoom_time: Optional[u.Quantity] = None, #30.0 * u.Myr,
        crop_ages: Optional[u.Quantity] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
        galfind_logger.debug(f"Plotting {repr(self)}")
        assert u.get_physical_type(time_units) == "time", \
            galfind_logger.critical(
                f"{time_units} with {u.get_physical_type(time_units)=}!='time'!"
            )
        if plot_type == "lookback":
            x = self.ages
        elif plot_type == "absolute":
            x = self.age_of_universe - self.ages
        else:
            err_message = f"{plot_type=} must be 'lookback' or 'absolute'!"
            galfind_logger.critical(err_message)
            raise ValueError(err_message)
        # convert ages to desired units
        x = x.to(time_units).value

        # calculate median and confidence interval for SFH posterior
        assert self.sfh_post.shape[1] == len(self.ages), \
            galfind_logger.critical(
                f"SFH posterior shape {self.sfh.shape=} does not match ages length {len(self.ages)=}!"
            )
        if crop_ages is not None:
            assert u.get_physical_type(crop_ages) == "time", \
                galfind_logger.critical(
                    f"{crop_ages} with {u.get_physical_type(crop_ages)=}!='time'!"
                )
            assert len(crop_ages) == 2, \
                galfind_logger.critical(
                    f"{crop_ages=} must be a list of two elements!"
                )
            crop_ages = crop_ages.to(time_units)
            crop_mask = (
                (self.ages.to(time_units) > crop_ages[0]) & 
                (self.ages.to(time_units) < crop_ages[1])
            )
            x = x[crop_mask]
            sfh_post = self.sfh_post[:, crop_mask]
        else:
            sfh_post = self.sfh_post
        sfh_post = np.percentile(sfh_post, (16, 50, 84), axis=0).T

        if "color" not in plot_kwargs.keys():
            plot_kwargs["color"] = primary_colour
        ax.plot(
            x,
            sfh_post[:, 1],
            **plot_kwargs
        )

        fill_between_kwargs = deepcopy(plot_kwargs)
        fill_between_kwargs["color"] = secondary_colour
        fill_between_kwargs["alpha"] = 0.5 * plot_kwargs.get("alpha", 1.0)
        ax.fill_between(
            x,
            sfh_post[:, 0],
            sfh_post[:, 2],
            **fill_between_kwargs
        )

        if annotate:
            if plot_type == "lookback":
                x_label = f"Lookback Time / {time_units}"
                if ax.get_xlim()[0] < 0:
                    ax.set_xlim(0, self.age_of_universe.to(time_units).value * 1.1)
                else:
                    ax.set_xlim(0, np.max([self.age_of_universe.to(time_units).value * 1.1, ax.get_xlim()[1]]))
                ax.set_ylim(1e-2, np.max([ax.get_ylim()[1], 1.1 * np.max(sfh_post[:, 2])]))
            elif plot_type == "absolute":
                x_label = f"Age of Universe / {time_units}"
                ax.set_xlim(self.age_of_universe.to(time_units).value, 0)
                ax.set_ylim(0., np.max([ax.get_ylim()[1], 1.1*np.max(sfh_post[:, 2])]))
                if z_axis:
                    import bagpipes as pipes
                    Gyr_ticks = np.interp(zvals, pipes.utils.z_array, pipes.utils.age_at_z) * u.Gyr
                    ticks = Gyr_ticks.to(time_units).value
                    z_ax = ax.twiny()
                    z_ax.set_xticks(ticks)
                    z_ax.set_xticklabels([rf"${z}$" for z in zvals])
                    z_ax.set_xlim(ax.get_xlim())
                    z_ax.set_xlabel(r"Redshift, $z$")
                    z_ax.grid(False)
            ax.set_xlabel(x_label)
            ax.set_ylabel(r"SFR / $\mathrm{{M}}_{{\odot}} \mathrm{{yr}}^{{-1}}$")
            if label_z:
                ax.text(
                    0.1,
                    0.9,
                    rf"$z_{{\mathrm{{obs}}}}={self.z:.2f}$",
                    transform = ax.transAxes,
                    ha = "left",
                    va = "top",
                    fontweight = "bold",
                )
            ax.grid(False)
        
        if zoom_time is not None:
            assert u.get_physical_type(zoom_time) == "time", \
                galfind_logger.critical(
                    f"{zoom_time=} with {u.get_physical_type(zoom_time)=}!='time'!"
                )
            assert isinstance(zoom_time.value, float), \
                galfind_logger.critical(
                    f"{zoom_time.value=} must be a float, got {type(zoom_time.value)=}!"
                )
            # make inset axis zooming in to most recent 'zoom_time' years
            zoom_time = zoom_time.to(time_units)
            zoom_ax = inset_axes(
                ax,
                width = "50%",
                height = "50%",
                loc='upper right',
            )
            self.plot(
                zoom_ax,
                time_units = time_units,
                crop_ages = [0.0, zoom_time.value] * zoom_time.unit,
                plot_type = "lookback",
                label_z = False,
                zoom_time = None,
                save = None,
                **plot_kwargs
            )
            zoom_ax.set_xlim(0.0, zoom_time.value)
            # halve the fontsize of x and y labels
            zoom_ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * 0.5)
            zoom_ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * 0.5)
            # # halve x and y tick label sizes
            # for label in zoom_ax.get_xticklabels():
            #     label.set_fontsize(label.get_fontsize() * 0.5)
            # for label in zoom_ax.get_yticklabels():
            #     label.set_fontsize(label.get_yticklabels()[0].get_fontsize() * 0.5)
            # halve x and y ticks sizes
            zoom_ax.tick_params(axis='x', which='major', labelsize=zoom_ax.get_xticklabels()[0].get_fontsize() * 0.5)
            zoom_ax.tick_params(axis='y', which='major', labelsize=zoom_ax.get_yticklabels()[0].get_fontsize() * 0.5)

            # Connect inset to main plot
            # Add rectangle on the main plot
            rect = Rectangle((ax.get_xlim()[0] - zoom_ax.get_xlim()[1], 0.0), zoom_ax.get_xlim()[1], zoom_ax.get_ylim()[1],
                 edgecolor='purple', linewidth=1, linestyle='--', facecolor='none')
            ax.add_patch(rect)
            # Draw lines connecting the inset to the main plot
            # Connect rectangle corners to inset (zoom_ax)
            # Connection from upper-left
            x0, x1 = zoom_ax.get_xlim()
            y0, y1 = zoom_ax.get_ylim()
            rect_x = ax.get_xlim()[0] - x1
            con1 = ConnectionPatch(xyA=(x0, y1), coordsA=zoom_ax.transData,
                                xyB=(rect_x + x1, y1), coordsB=ax.transData,
                                color='purple', linestyle='--', linewidth=1)
            # Connection from lower-right
            con2 = ConnectionPatch(xyA=(x1, y0), coordsA=zoom_ax.transData,
                                xyB=(rect_x, y0), coordsB=ax.transData,
                                color='purple', linestyle='--', linewidth=1)

            # Add connections to the figure
            ax.add_artist(con1)
            ax.add_artist(con2)
            # #mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.5")
        else:
            zoom_ax = None
            
        if save:
            err_message = "Saving SFH plot is not implemented yet!"
            galfind_logger.warning(err_message)
            raise NotImplementedError(err_message)
        
        if not (plot_type == "absolute" and z_axis):
            z_ax = None
        return ax, z_ax, zoom_ax
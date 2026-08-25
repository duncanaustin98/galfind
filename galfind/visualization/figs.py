"""Matplotlib figure and axis layout utilities.

Provides functions for creating rectangular and square axis grids with
automatic scaling and layout configuration.
"""

from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ..utils.exceptions import GalfindTypeError, RangeError


def make_rectangular_fig(
    n_ax: int,
    xy_ratio: Union[int, float],
    scaling: float = 3.0,
    axis_type: str = "cutout",
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a figure with a rectangular grid of axes.

    Automatically arranges ``n_ax`` subplots in a grid with the specified
    aspect ratio, sizing each axis based on the scaling factor.

    Parameters
    ----------
    n_ax : `int`
        Number of subplots (axes) to create.
    xy_ratio : `int` or `float`
        Aspect ratio (width:height) for the grid layout.
    scaling : `float`, optional
        Size scaling for each axis in inches. Default is 3.0.
    axis_type : `str`, optional
        Type of axes: "cutout" (no ticks/labels) or standard axes.
        Default is "cutout".
    sharex : `bool`, optional
        Share x-axis across axes. Default is `False`.
    sharey : `bool`, optional
        Share y-axis across axes. Default is `False`.
    **gridspec_kwargs
        Additional keyword arguments passed to `fig.add_gridspec()`.

    Returns
    -------
    `tuple` of (`matplotlib.figure.Figure`, `list` of `matplotlib.axes.Axes`)
        The figure and list of axes objects.
    """
    if not isinstance(n_ax, int):
        raise GalfindTypeError(
            f"n_ax={n_ax!r} has type {type(n_ax).__name__}; must be int."
        )
    if n_ax <= 0:
        raise RangeError(f"n_ax={n_ax} must be > 0.")
    n_x = int(np.ceil(np.sqrt(n_ax * xy_ratio)))
    n_y = int(np.ceil(n_ax / n_x))
    return make_fig_ax(
        n_x,
        n_y,
        scaling=scaling,
        axis_type=axis_type,
        sharex=sharex,
        sharey=sharey,
        **gridspec_kwargs,
    )


def make_square_fig(
    n_ax: int,
    scaling: float = 3.0,
    axis_type: str = "cutout",
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a figure with a square grid of axes.

    Arranges ``n_ax`` subplots in an N×N grid (where N = √n_ax).

    Parameters
    ----------
    n_ax : `int`
        Number of subplots, must be a perfect square.
    scaling : `float`, optional
        Size scaling for each axis in inches. Default is 3.0.
    axis_type : `str`, optional
        Type of axes: "cutout" (no ticks/labels) or standard axes.
        Default is "cutout".
    sharex : `bool`, optional
        Share x-axis across axes. Default is `False`.
    sharey : `bool`, optional
        Share y-axis across axes. Default is `False`.
    **gridspec_kwargs
        Additional keyword arguments passed to `fig.add_gridspec()`.

    Returns
    -------
    `tuple` of (`matplotlib.figure.Figure`, `list` of `matplotlib.axes.Axes`)
        The figure and list of axes objects.

    Raises
    ------
    GalfindTypeError
        If ``n_ax`` is not a perfect square.
    """
    # NOTE: `np.sqrt(n_ax)` always returns a `numpy.float64`, so this
    # `isinstance(..., int)` check can never pass -- pre-existing no-op
    # bug preserved as-is (not this conversion's job to fix the
    # underlying condition, only the exception identity/message).
    if not isinstance(np.sqrt(n_ax), int):
        raise GalfindTypeError(
            f"n_ax={n_ax!r} does not yield an integer sqrt "
            f"(np.sqrt(n_ax)={np.sqrt(n_ax)!r}); n_ax must be a perfect "
            "square."
        )
    n_x = int(np.sqrt(n_ax))
    return make_fig_ax(
        n_x,
        n_x,
        scaling=scaling,
        axis_type=axis_type,
        sharex=sharex,
        sharey=sharey,
        **gridspec_kwargs,
    )


def make_fig_ax(
    n_x: int,
    n_y: int,
    scaling: float = 3.0,
    axis_type: str = "cutout",
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a figure with a specified grid of axes.

    Creates a figure and populates it with a grid of subplots with the
    specified dimensions.

    Parameters
    ----------
    n_x : `int`
        Number of columns in the grid.
    n_y : `int`
        Number of rows in the grid.
    scaling : `float`, optional
        Size scaling for each axis in inches. Default is 3.0.
    axis_type : `str`, optional
        Type of axes: "cutout" (no ticks/labels) or standard axes.
        Default is "cutout".
    sharex : `bool`, optional
        Share x-axis across axes. Default is `False`.
    sharey : `bool`, optional
        Share y-axis across axes. Default is `False`.
    **gridspec_kwargs
        Additional keyword arguments passed to `fig.add_gridspec()`.

    Returns
    -------
    `tuple` of (`matplotlib.figure.Figure`, `list` of `matplotlib.axes.Axes`)
        The figure and list of axes objects.
    """
    fig = make_fig(n_x, n_y, scaling)
    if axis_type == "cutout":
        ax_arr = make_cutout_ax(
            fig, n_x, n_y, **gridspec_kwargs
        )  # , sharex = sharex, sharey = sharey)
    else:
        ax_arr = make_ax(
            fig, n_x, n_y, sharex=sharex, sharey=sharey, **gridspec_kwargs
        )
    return fig, ax_arr


def make_fig(
    n_x: int,
    n_y: int,
    scaling: float = 3.0,
) -> plt.Figure:
    """Create a matplotlib figure with specified dimensions.

    Parameters
    ----------
    n_x : `int`
        Number of columns (used to compute width).
    n_y : `int`
        Number of rows (used to compute height).
    scaling : `float`, optional
        Size of each "unit" in inches. Figure size = (n_x, n_y) × scaling.
        Default is 3.0.

    Returns
    -------
    `matplotlib.figure.Figure`
        A new figure object.
    """
    return plt.figure(figsize=(n_x * scaling, n_y * scaling))


def make_ax(
    fig: plt.Figure,
    n_x: int,
    n_y: int,
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> List[plt.Axes]:
    """Create a grid of standard matplotlib axes in a figure.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        Figure to add axes to.
    n_x : `int`
        Number of columns in the grid.
    n_y : `int`
        Number of rows in the grid.
    sharex : `bool`, optional
        Share x-axis among all axes. Default is `False`.
    sharey : `bool`, optional
        Share y-axis among all axes. Default is `False`.
    **gridspec_kwargs
        Additional keyword arguments passed to `fig.add_gridspec()`.

    Returns
    -------
    `list` of `matplotlib.axes.Axes`
        List of axes objects in row-major order.
    """
    gridspec_cutout = fig.add_gridspec(n_y, n_x, **gridspec_kwargs)
    cutout_ax_list = []
    for i in range(n_x * n_y):
        if i == 0:
            cutout_ax = fig.add_subplot(gridspec_cutout[i])
        else:
            cutout_ax = fig.add_subplot(
                gridspec_cutout[i],
                sharex=cutout_ax_list[0] if sharex else None,
                sharey=cutout_ax_list[0] if sharey else None,
            )
        cutout_ax_list.extend([cutout_ax])
    ax_arr = np.array(cutout_ax_list, dtype=object).flatten()
    return ax_arr


def make_cutout_ax(
    fig: plt.Figure,
    n_x: int,
    n_y: int,
    **gridspec_kwargs: Dict[str, Any],
) -> List[plt.Axes]:
    """Create a grid of image cutout axes with equal aspect ratio and no ticks.

    Generates axes suitable for displaying image cutouts, with automatic
    equal aspect ratio, no axis ticks or labels.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        Figure to add axes to.
    n_x : `int`
        Number of columns in the grid.
    n_y : `int`
        Number of rows in the grid.
    **gridspec_kwargs
        Additional keyword arguments passed to `fig.add_gridspec()`.

    Returns
    -------
    `list` of `matplotlib.axes.Axes`
        List of cutout axes objects in row-major order.
    """
    gridspec_cutout = fig.add_gridspec(n_y, n_x, **gridspec_kwargs)
    cutout_ax_list = []
    for i in range(n_x * n_y):
        cutout_ax = fig.add_subplot(gridspec_cutout[i])
        cutout_ax.set_aspect("equal", adjustable="box", anchor="S")
        cutout_ax.set_xticks([])
        cutout_ax.set_yticks([])
        cutout_ax_list.extend([cutout_ax])
    ax_arr = np.array(cutout_ax_list, dtype=object).flatten()
    return ax_arr


def make_phot_diagnostic_fig(
    n_cutouts: int,
    fig_kwargs: Dict[str, Any] = {},
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a diagnostic figure layout for photometry visualization.

    Creates a complex multi-panel figure with separate subfigures for
    photometry plots and image cutouts, automatically adjusting height
    ratios based on the number of cutouts.

    Parameters
    ----------
    n_cutouts : `int`
        Number of cutout images to display.
    fig_kwargs : `dict`, optional
        Keyword arguments passed to `plt.figure()`. Common keys include
        "figsize", "constrained_layout", etc. Default is `{}`.

    Returns
    -------
    `tuple` of (`matplotlib.figure.Figure`, `list`)
        Figure containing:
        - overall_fig: The top-level figure
        - fig_axs: List containing [cutout_subfig, phot_ax, PDF_ax_list]
    """
    # figure size may well depend on how many bands there are
    fig_kwargs.setdefault("figsize", (8.0, 7.0))
    fig_kwargs.setdefault("constrained_layout", True)
    overall_fig = plt.figure(**fig_kwargs)
    fig, cutout_fig = overall_fig.subfigures(
        2,
        1,
        hspace=-2.0,
        height_ratios=[2.0, 1.0] if n_cutouts <= 8 else [1.8, 1],
    )

    gs = fig.add_gridspec(2, 4)
    phot_ax = fig.add_subplot(gs[:, 0:3])

    PDF_ax = [fig.add_subplot(gs[0, 3:]), fig.add_subplot(gs[1, 3:])]

    fig_axs = [cutout_fig, phot_ax, PDF_ax]

    return overall_fig, fig_axs

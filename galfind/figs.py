import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Any


def make_rectangular_fig(
    n_ax: int,
    xy_ratio: Union[int, float],
    scaling: float = 3.0,
    axis_type: str = "cutout",
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    assert isinstance(n_ax, int) and n_ax > 0
    n_x = int(np.ceil(np.sqrt(n_ax * xy_ratio)))
    n_y = int(np.ceil(n_ax / n_x))
    return make_fig_ax(n_x, n_y, scaling = scaling, axis_type = axis_type, sharex = sharex, sharey = sharey, **gridspec_kwargs)


def make_square_fig(
    n_ax: int,
    scaling: float = 3.0,
    axis_type: str = "cutout",
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    assert isinstance(np.sqrt(n_ax), int)
    n_x = int(np.sqrt(n_ax))
    return make_fig_ax(n_x, n_x, scaling = scaling, axis_type = axis_type, sharex = sharex, sharey = sharey, **gridspec_kwargs)


def make_fig_ax(
    n_x: int,
    n_y: int,
    scaling: float = 3.0,
    axis_type: str = "cutout",
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    fig = make_fig(n_x, n_y, scaling)
    if axis_type == "cutout":
        ax_arr = make_cutout_ax(fig, n_x, n_y, **gridspec_kwargs) #, sharex = sharex, sharey = sharey)
    else:
        ax_arr = make_ax(fig, n_x, n_y, sharex = sharex, sharey = sharey, **gridspec_kwargs)
    return fig, ax_arr


def make_fig(
    n_x: int,
    n_y: int,
    scaling: float = 3.0,
) -> plt.Figure:
    return plt.figure(figsize=(n_x * scaling, n_y * scaling))

def make_ax(
    fig: plt.Figure,
    n_x: int,
    n_y: int,
    sharex: bool = False,
    sharey: bool = False,
    **gridspec_kwargs: Dict[str, Any],
) -> List[plt.Axes]:
    gridspec_cutout = fig.add_gridspec(n_y, n_x, **gridspec_kwargs)
    cutout_ax_list = []
    for i in range(n_x * n_y):
        if i == 0:
            cutout_ax = fig.add_subplot(gridspec_cutout[i])
        else:
            cutout_ax = fig.add_subplot(gridspec_cutout[i], sharex=cutout_ax_list[0] if sharex else None, sharey=cutout_ax_list[0] if sharey else None)
        cutout_ax_list.extend([cutout_ax])
    ax_arr = np.array(cutout_ax_list, dtype=object).flatten()
    return ax_arr

def make_cutout_ax(
    fig: plt.Figure,
    n_x: int,
    n_y: int,
    **gridspec_kwargs: Dict[str, Any],
) -> List[plt.Axes]:
    gridspec_cutout = fig.add_gridspec(n_y, n_x)
    cutout_ax_list = []
    for i in range(n_x * n_y):
        cutout_ax = fig.add_subplot(gridspec_cutout[i])
        cutout_ax.set_aspect("equal", adjustable="box", anchor="N")
        cutout_ax.set_xticks([])
        cutout_ax.set_yticks([])
        cutout_ax_list.extend([cutout_ax])
    ax_arr = np.array(cutout_ax_list, dtype=object).flatten()
    return ax_arr

def make_phot_diagnostic_fig(n_cutouts: int):
    # figure size may well depend on how many bands there are
    overall_fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    fig, cutout_fig = overall_fig.subfigures(
        2,
        1,
        hspace = -2.0,
        height_ratios = [2.0, 1.0]
        if n_cutouts <= 8
        else [1.8, 1],
    )

    gs = fig.add_gridspec(2, 4)
    phot_ax = fig.add_subplot(gs[:, 0:3])

    PDF_ax = [fig.add_subplot(gs[0, 3:]), fig.add_subplot(gs[1, 3:])]

    fig_axs = [cutout_fig, phot_ax, PDF_ax]

    return overall_fig, fig_axs

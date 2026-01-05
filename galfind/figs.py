import numpy as np
import matplotlib.pyplot as plt


def make_rectangular_fig(n_ax, xy_ratio):
    pass


def make_square_fig(n_ax):
    assert isinstance(np.sqrt(n_ax), int)
    pass


def make_fig_ax(n_x: int, n_y: int, scaling: float = 3.):
    fig = make_fig(n_x, n_y, scaling)
    ax_arr = make_ax(fig, n_x, n_y)
    return fig, ax_arr

def make_fig(n_x: int, n_y: int, scaling: float = 3.):
    return plt.figure(figsize=(n_x * scaling, n_y * scaling))

def make_ax(fig: plt.Figure, n_x: int, n_y: int):
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
        hspace=-2.,
        height_ratios=[2.0, 1.0]
        if n_cutouts <= 8
        else [1.8, 1],
    )

    gs = fig.add_gridspec(2, 4)
    phot_ax = fig.add_subplot(gs[:, 0:3])

    PDF_ax = [fig.add_subplot(gs[0, 3:]), fig.add_subplot(gs[1, 3:])]

    fig_axs = [cutout_fig, phot_ax, PDF_ax]

    return fig_axs

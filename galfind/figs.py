import numpy as np
import matplotlib.pyplot as plt


def make_rectangular_fig(n_ax, xy_ratio):
    pass


def make_square_fig(n_ax):
    assert isinstance(np.sqrt(n_ax), int)


def make_ax(fig: plt.Figure, n_x: int, n_y: int):
    gridspec_cutout = fig.add_gridspec(n_y, n_x)
    cutout_ax_list = []
    for i in range(len(n_x * n_y)):
        cutout_ax = fig.add_subplot(gridspec_cutout[i])
        cutout_ax.set_aspect("equal", adjustable="box", anchor="N")
        cutout_ax.set_xticks([])
        cutout_ax.set_yticks([])
        cutout_ax_list.extend([cutout_ax])
    ax_arr = np.array(cutout_ax_list, dtype=object).flatten()
    return ax_arr

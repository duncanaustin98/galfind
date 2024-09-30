# Depths.py

# import automask as am
import astropy.visualization as vis
import cv2 as cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import h5py
import astropy.units as u
from astropy.coordinates import SkyCoord
from copy import deepcopy
from pathlib import Path
from astropy.io import fits
from astropy.table import Table, vstack
import os
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numba import jit
from photutils.aperture import CircularAperture
from scipy.stats import gaussian_kde
from skimage import morphology
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Union, Tuple, Dict, List, Any, NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Band_Data, Band_Data_Base, Data

try:
    from typing import Type  # python 3.11+
except ImportError:
    from typing_extensions import Type  # python > 3.7 AND python < 3.11

# install cv2, skimage, sklearn
from . import useful_funcs_austind as funcs
from . import config, galfind_logger


def do_photometry(image, xy_coords, radius_pixels):
    aper = CircularAperture(xy_coords, radius_pixels)
    aper_sums, _ = aper.do_photometry(image, error=None)
    return aper_sums


def make_grid(
    data,
    mask,
    radius,
    scatter_size,
    pixel_scale=0.03,
    plot=False,
    ax=None,
    distance_to_mask=50,
):
    """
    data: 2D numpy array
    radius: float in arcseconds
    scatter_size: float in arcseconds
    pixel_scale: float in arcseconds/pixel
    """
    radius_pixels = radius / pixel_scale
    scatter_size_pixels = scatter_size / pixel_scale

    # assert radius > scatter_size, "Radius must be greater than scatter size"
    # NOTE!!!!
    # np.shape on a 2D array returns (y, x) not (x, y)
    # So references to an x, y coordinate in the array should be [y, x]

    xy = np.mgrid[
        radius_pixels + scatter_size_pixels : data.shape[1]
        - (radius_pixels + scatter_size_pixels) : 2
        * (radius_pixels + scatter_size_pixels),
        radius_pixels + scatter_size_pixels : data.shape[0]
        - (radius_pixels + scatter_size_pixels) : 2
        * (radius_pixels + scatter_size_pixels),
    ]

    xy = xy.reshape(2, -1).T

    scatter = np.random.uniform(
        low=-scatter_size_pixels,
        high=scatter_size_pixels,
        size=(xy.shape[0], 2),
    )

    xy = xy + scatter
    # print('Img shape, mask shape, max x grid, max y grid')
    # print(data.shape, mask.shape, np.max(xy[:, 0]), np.max(xy[:, 1]))

    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (distance_to_mask, distance_to_mask)
    )  # set up a circle of radius distance_to_mask pixels to mask around location of 0's
    mask = cv2.dilate(
        mask, kernel, iterations=1
    )  # dilate mask using the circle

    mask = mask.astype(bool)

    non_overlapping_xy = []
    for x, y in xy:
        # Check if the circle overlaps with the mask
        if np.any(
            mask[
                int(y - radius_pixels) : int(y + radius_pixels),
                int(x - radius_pixels) : int(x + radius_pixels),
            ]
        ):
            continue  # Skip this coordinate if it overlaps
        else:
            non_overlapping_xy.append(
                (x, y)
            )  # Add non-overlapping coordinates to the list
    # Plot the circles using matplotlib
    # if plot:
    #     if ax == None:
    #         fig, ax = plt.subplots()

    #     stretch = vis.CompositeStretch(
    #         vis.LogStretch(), vis.ContrastBiasStretch(contrast=30, bias=0.08)
    #     )
    #     norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)

    #     ax.imshow(
    #         data, cmap="Greys", origin="lower", interpolation="None", norm=norm
    #     )

    #     for x, y in non_overlapping_xy:
    #         circle = plt.Circle((x, y), radius_pixels, color="r", fill=True)
    #         ax.add_artist(circle)

    return non_overlapping_xy


def show_depths(
    nmad_grid,
    num_grid,
    step_size,
    region_radius_used_pix,
    labels=None,
    cat_labels=None,
    cat_depths=None,
    cat_diagnostics=None,
    x_pix=None,
    y_pix=None,
    img_mask=None,
    labels_final=None,
    suptitle=None,
    save_path=None,
    show=False,
):
    fig, axs = plt.subplots(
        3 if type(labels) == type(None) else 4,
        1 if type(cat_depths) == type(None) else 2,
        facecolor="white",
        figsize=(16, 16),
        constrained_layout=True,
    )

    # move axis apart
    axs = axs.flatten()
    if type(suptitle) != type(None):
        fig.suptitle(suptitle, fontsize="large", fontweight="bold")
    cmap = cm.get_cmap("plasma")
    cmap.set_bad(color="black")
    nmad_grid[nmad_grid == 0] = np.nan
    # Make vmin and vmax the 5th and 95th percentile of the nmad_grid
    vmin, vmax = (
        np.nanpercentile(nmad_grid, 1),
        np.nanpercentile(nmad_grid, 99),
    )
    mappable = axs[0].imshow(
        nmad_grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
    )
    radius = region_radius_used_pix / step_size
    circle_x = radius + 0.07 * np.shape(nmad_grid)[1]
    circle_y = radius + 0.07 * np.shape(nmad_grid)[0]

    print("Circle x, y:", circle_x, circle_y)
    patch = plt.Circle(
        (circle_x, circle_y),
        radius=radius,
        fill=True,
        facecolor="white",
        lw=2,
        zorder=10,
    )
    axs[0].add_patch(patch)
    axs[0].text(
        circle_x,
        circle_y - 1.4 * radius,
        s="Filter Size",
        va="center",
        ha="center",
        color="white",
        fontsize="medium",
    )  # path_effects = [pe.Stroke(linewidth=0.5, foreground='black')])
    # Make colorbar same height as ax
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mappable, label=r"5$\sigma$ Depth", cax=cax)

    axs[0].set_title(r"Rolling Average 5$\sigma$ Depth")

    mappable2 = axs[1].imshow(num_grid, origin="lower", cmap=cmap)
    divider2 = make_axes_locatable(axs[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(mappable2, label=r"Number of Apertures Used", cax=cax2)
    axs[1].set_title("Rolling Average Diagnostic")

    if type(labels) != type(None):
        cmap = cm.get_cmap("Set2")

        possible_labels = np.unique(labels)
        av_depths = [
            np.nanmedian(nmad_grid[labels == label])
            for label in possible_labels
        ]
        # mask nan in av_depths
        av_depths = np.array(av_depths)[~np.isnan(av_depths)]
        if len(av_depths) > 1:
            num_labels = len(np.unique(labels))
            custom_cmap = LinearSegmentedColormap.from_list(
                "custom",
                [cmap(i / num_labels) for i in range(num_labels)],
                num_labels,
            )

            pos = 5 if type(cat_depths) != type(None) else 2
            mappable = axs[pos + 1].imshow(
                labels, cmap=custom_cmap, origin="lower", interpolation="None"
            )

            possible_labels = np.unique(labels)

            order = np.argsort(av_depths)
            possible_labels = possible_labels[order]

            labels_arr = ["Shallow", "Deep"]
            colours = [
                custom_cmap(possible_labels[0]),
                custom_cmap(possible_labels[1]),
            ]

            axs[pos + 1].set_title("Labels")
            axs[pos].set_title("Catalogue Labels")

            axs[pos].imshow(
                img_mask,
                cmap="Greens",
                origin="lower",
                interpolation="None",
                alpha=0.3,
                zorder=4,
            )
            axs[pos].imshow(
                labels_final,
                cmap="Reds",
                origin="lower",
                interpolation="None",
                alpha=0.3,
                zorder=4,
            )
            m = axs[pos].scatter(
                x_pix,
                y_pix,
                s=1,
                zorder=5,
                c=np.array(cat_labels),
                cmap="plasma",
            )
            # fig.colorbar(ax=axs[5], mappable=m, label='Label')
        else:
            labels_arr = ["Single Region"]
            possible_labels = [0]
            colours = [cmap(0)]
            axs[-2].remove()
            axs[-1].remove()

    # Histogram of depths
    if type(cat_depths) != type(None):
        # plt.scatter(x_pix, y_pix, s=1, zorder=5, c = depth, cmap='plasma')

        axs[2].set_title("Catalogue Depths")
        m = axs[2].scatter(
            x_pix, y_pix, s=1, zorder=5, c=cat_depths, cmap="plasma"
        )
        divider3 = make_axes_locatable(axs[2])
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable=m, label=r"5$\sigma$ Depth", cax=cax3)

        axs[2].imshow(
            img_mask,
            cmap="Greens",
            origin="lower",
            interpolation="None",
            alpha=0.3,
            zorder=4,
        )
        axs[2].imshow(
            labels_final,
            cmap="Reds",
            origin="lower",
            interpolation="None",
            alpha=0.3,
            zorder=4,
        )

        axs[4].set_title("Catalogue Diagnostic")
        m = axs[4].scatter(
            x_pix,
            y_pix,
            s=1,
            zorder=5,
            c=np.array(cat_diagnostics),
            cmap="plasma",
        )
        # Make it so axes aren't stetch disproportionately
        axs[4].set_aspect("equal")
        divider4 = make_axes_locatable(axs[4])
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(
            mappable=m, label="Distance to 200th Empty Aperture", cax=cax4
        )

        set_labels = [
            cat_depths[cat_labels == label] for label in possible_labels
        ]
        for set_label, label, colour in zip(set_labels, labels_arr, colours):
            _set_label = set_label[~np.isnan(set_label)]  # remove nans
            axs[3].hist(
                _set_label,
                bins=40,
                range=(np.nanmin(cat_depths), np.nanmax(cat_depths)),
                label=label,
                color=colour,
                histtype="stepfilled",
                alpha=0.8,
            )
        # Plot line at median depth
        # Fix y range
        axs[3].set_ylim(axs[3].get_ylim())
        max = axs[3].get_ylim()[1]

        for pos, (depth, colour) in enumerate(zip(set_labels, colours)):
            axs[3].axvline(
                np.nanmedian(depth),
                0,
                max,
                color="black",
                lw=3,
                linestyle="--",
                label="Median" if pos == 0 else "",
                zorder=10,
            )
            axs[3].axvline(
                np.nanmean(depth),
                0,
                max,
                color="black",
                lw=3,
                linestyle="-",
                label="Mean" if pos == 0 else "",
                zorder=10,
            )
            # Label with text
            axs[3].text(
                np.nanmean(depth),
                0.7 * max,
                f"{np.nanmean(depth):.2f}",
                va="top",
                ha="center",
                fontsize="medium",
                color=colour,
                rotation=90,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                zorder=10,
                fontweight="bold",
            )
            axs[3].text(
                np.nanmedian(depth),
                0.9 * max,
                f"{np.nanmedian(depth):.2f}",
                va="top",
                ha="center",
                fontsize="medium",
                color=colour,
                rotation=90,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                zorder=10,
                fontweight="bold",
            )

        axs[3].set_xlabel(r"5$\sigma$ Depth")
        axs[3].set_title("Depth Histogram")
        axs[3].legend(frameon=False)

    else:
        print("No catalogue depths")
    # axs[7].remove()
    # plt.tight_layout()
    # Delete all axes with nothing on
    for ax in axs:
        if (
            len(ax.images) == 0
            and len(ax.collections) == 0
            and len(ax.lines) == 0
            and len(ax.patches) == 0
            and len(ax.artists) == 0
        ):
            try:
                ax.remove()
            except:
                pass

    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)

    if type(save_path) != type(None):
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        funcs.change_file_permissions(save_path)
        print(f"Saved depths plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.clf()

    print("Median 5 sigma depth:", np.nanmedian(nmad_grid))
    print("Median number of apertures used:", np.nanmedian(num_grid))
    print("Mean 5 sigma depth:", np.nanmean(nmad_grid))
    return fig, axs


def calc_depths(
    coordinates,
    fluxes,
    img_data,
    mask=None,
    catalogue=None,
    mode="rolling",
    sigma_level=5,
    step_size=100,
    region_radius_used_pix=300,
    zero_point=28.08,
    min_number_of_values=100,
    n_nearest=100,
    split_depths=False,
    wht_data=None,
    n_split=1,
    split_depth_min_size=100000,
    split_depths_factor=5,
    coord_type="sky",
    wcs=None,
    provide_labels=None,
    diagnostic_id=None,
    plot=False,
):
    """
    coordinates: list of tuples - (x, y) coordinates
    fluxes: list of floats - fluxes corresponding to the coordinates
    img_data: 2D numpy array - the image data
    mask: 2D numpy array - the mask
    mode: str - 'rolling' or 'n_nearest'
    sigma_level: float - the number of sigmas to use for the depth calculation
    step_size: int - the number of pixels to subgrid the image
    region_radius_used_pix: int - the radius of the window in pixels - only used if mode is 'rolling'
    zero_point: float - the zero point for the depth calculation
    min_number_of_values: int - the minimum number of values required to calculate the depth if the mode is 'rolling'
    n_nearest: int - the number of nearest neighbors to use for the depth calculation - only used if mode is 'n_nearest'
    n_split: int - the number of regions to split the depths into using KMeans clustering
    split_depth_min_size: int - the minimum size of the regions
    split_depths_factor - int - the factor to use for the binning of the weight map
    wht_data: 2D numpy array - the weight data - only used if split_depths is True
    coord_type: str - 'sky' or 'pixel'
    wcs: WCS object - the wcs object to use for the conversion if coord_type is 'sky'
    diagnostic_id: int - the position of a galaxy in the catalogue to show the diagnostic plot
    plot: bool - whether the nmad grid is plotted or not
    """
    print("This is the experimental numba version")
    # Determine whether to split depths or not
    if n_split == 1:
        split_depths = False
    else:
        split_depths = True
    # Extract x and y coordinates
    coordinates = np.array(coordinates)
    x, y = coordinates[:, 0], coordinates[:, 1]

    # if type(wht_data) == str:
    #     weight_map = fits.open(wht_data)
    #     print(f"Opening weight map: {wht_data}")
    #     # Check if we have multiple extensions
    #     if len(weight_map) > 1:
    #         weight_map = weight_map['WHT'].data
    #     else:
    #         weight_map = weight_map[0].data
    #     wht_data = weight_map

    # Determine the grid size
    if wht_data is not None and split_depths:
        if provide_labels is None:
            print("Obtaining labels...")
            assert (
                np.shape(wht_data) == np.shape(img_data)
            ), f"The weight map must have the same shape as the image {np.shape(wht_data)} != {np.shape(img_data)}"
            labels_final, weight_map_smoothed = cluster_wht_map(
                wht_data,
                num_regions=n_split,
                bin_factor=split_depths_factor,
                min_size=split_depth_min_size,
            )
            print("Labels obtained")
        else:
            labels_final = provide_labels
            print("Using provided labels")
    else:
        print("Not labelling data")
        labels_final = np.zeros_like(img_data, dtype=np.float64)

    #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
    # So references to an x, y coordinate in the array should be [y, x]

    if catalogue is None:
        # If the catalogue is not provided, use the grid mode
        iterate_mode = "grid"
    else:
        # If the catalogue is provided, use the catalogue mode
        iterate_mode = "catalogue"
        # Correct the coordinates if they are in sky coordinates

        if coord_type == "sky" and wcs is not None:
            # This doesn't work because footprint of the image is not the same as the footprint of the catalogue
            cat_x_col, cat_y_col = "ALPHA_J2000", "DELTA_J2000"
            ra_pix, dec_pix = wcs.all_world2pix(
                catalogue[cat_x_col], catalogue[cat_y_col], 0
            )
            cat_x, cat_y = ra_pix, dec_pix
            if wht_data is not None:
                assert np.shape(wht_data) == np.shape(img_data)

        elif coord_type == "pixel":
            cat_x_col, cat_y_col = "X_IMAGE", "Y_IMAGE"
            cat_x, cat_y = catalogue[cat_x_col], catalogue[cat_y_col]
        else:
            raise ValueError('coord_type must be either "sky" or "pixel"')

    x_max, y_max = np.max(x), np.max(y)
    x_label, y_label = x.astype(int), y.astype(int)
    # Don't look for label of pixels outside the image

    x_label = np.clip(x_label, 0, x_max - 1).astype(int)
    y_label = np.clip(y_label, 0, y_max - 1).astype(int)

    filter_labels = labels_final[y_label, x_label]  # .astype(np.float64)

    # i is the x coordinate, j is the y coordinate
    print("Iterate mode:", iterate_mode)
    if iterate_mode == "grid":
        grid_size = (
            int(np.ceil(x_max)) + step_size,
            int(np.ceil(y_max)) + step_size,
        )

        nmad_sized_grid = np.zeros(
            (grid_size[1] // step_size + 1, grid_size[0] // step_size + 1)
        )
        num_sized_grid = np.zeros(
            (grid_size[1] // step_size + 1, grid_size[0] // step_size + 1)
        )
        num_sized_grid[:] = np.nan
        label_size_grid = np.zeros(
            (grid_size[1] // step_size + 1, grid_size[0] // step_size + 1)
        )
        label_size_grid[:] = np.nan
        # print('Grid size:', grid_size)
        for i in tqdm(range(0, grid_size[0], step_size)):
            for j in range(0, grid_size[1], step_size):
                setnan = False
                if mask is not None:
                    # Don't calculate the depth if the coordinate is masked
                    try:
                        #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
                        # So references to an x, y coordinate in the array should be [y, x]
                        if mask[j, i] == 1.0:
                            depth = np.nan
                            setnan = True
                            num_of_apers = np.nan
                    except IndexError:
                        # print('index error')
                        setnan = True
                        depth = np.nan
                        num_of_apers = np.nan

                if not setnan:
                    j_label = np.clip(j, 0, y_max - 1)
                    i_label = np.clip(i, 0, x_max - 1)
                    #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
                    # So references to an x, y coordinate in the array should be [y, x]
                    label = labels_final[
                        j_label.astype(int), i_label.astype(int)
                    ]
                    distances = numba_distances(x, y, i, j)

                    mask_arr = filter_labels == label
                    distances_i = distances[mask_arr]
                    fluxes_i = fluxes[mask_arr]

                    label_name = label
                    if mode == "rolling":
                        # Extract the neighboring Y values within the circular window
                        # Ensure label values of regions are the same as label
                        neighbor_values = fluxes_i[
                            (distances_i <= region_radius_used_pix)
                            & (filter_labels == label)
                        ]
                        # neighbor_values = fluxes[distances <= region_radius_used_pix]
                    elif mode == "n_nearest":
                        nearest_indices = np.argpartition(
                            distances_i, n_nearest
                        )[:n_nearest]
                        neighbor_values = fluxes_i[nearest_indices]
                        min_number_of_values = n_nearest

                    num_of_apers = len(neighbor_values)

                    depth = calculate_depth(
                        neighbor_values,
                        sigma_level,
                        zero_point,
                        min_number_of_values=min_number_of_values,
                    )

                    # NOTE np.shape on a 2D array returns (y, x) not (x, y)
                    # So references to an x, y coordinate in the array should be [y, x]

                    num_sized_grid[j // step_size, i // step_size] = (
                        num_of_apers
                    )
                    nmad_sized_grid[j // step_size, i // step_size] = depth
                    label_size_grid[j // step_size, i // step_size] = (
                        label_name
                    )

        # if plot:
        #     plt.imshow(
        #         nmad_sized_grid,
        #         origin="lower",
        #         interpolation="None",
        #         cmap="plasma",
        #     )
        #     plt.show()

        return nmad_sized_grid, num_sized_grid, label_size_grid, labels_final

    elif iterate_mode == "catalogue":
        depths, diagnostic, cat_labels = [], [], []
        count = 0
        print("Total number", len(cat_x))
        for i, j in tqdm(zip(cat_x, cat_y), total=len(cat_x)):
            # Check if the coordinate is outside the image or in the mask
            if i > x_max or i < 0 or j > y_max or j < 0:
                depth = np.nan
                num_of_apers = np.nan
                label = np.nan
                depth_diagnostic = np.nan
            else:
                label = labels_final[j.astype(int), i.astype(int)]

                # distances = np.sqrt((x - i)**2 + (y - j)**2)
                # NOTE np.shape on a 2D array returns (y, x) not (x, y)
                # So references to an x, y coordinate in the array should be [y, x]

                distances = numba_distances(x, y, i, j)
                # Get the label of interest
                label = labels_final[j.astype(int), i.astype(int)]

                # Create a boolean mask
                mask = filter_labels == label

                distances_i = distances[mask]
                fluxes_i = fluxes[mask]

                if mode == "rolling":
                    neighbor_values = fluxes_i[
                        (distances_i <= region_radius_used_pix)
                        & (labels_final[y_label, x_label] == label)
                    ]
                    depth_diagnostic = len(neighbor_values)

                elif mode == "n_nearest":
                    # print(labels_final.dtype, y_label.dtype, x_label.dtype, fluxes.dtype, n_nearest.dtype, label.dtype)
                    # neighbor_values, depth_diagnostic = numba_n_nearest_filter(fluxes_i, distances_i, n_nearest)
                    nearest_indices = np.argpartition(distances_i, n_nearest)[
                        :n_nearest
                    ]
                    neighbor_values = fluxes_i[nearest_indices]

                    min_number_of_values = n_nearest
                    # Depth diagnostic is distance to n_nearest
                    depth_diagnostic = n_nearest  # distances_i[np.argsort(distances_i[nearest_indices])][-1]

                    # if plot:
                    #     if count == diagnostic_id:
                    #         # Plot regions used and image
                    #         fig, ax = plt.subplots()
                    #         ax.imshow(
                    #             img_data,
                    #             cmap="Greys",
                    #             origin="lower",
                    #             interpolation="None",
                    #         )

                    #         # Do this with matplotlib instead
                    #         circle = plt.Circle(
                    #             (i, j), radius_pixels, color="r", fill=False
                    #         )
                    #         ax.add_artist(circle)
                    #         xtest, ytest = (
                    #             x[labels_final[y_label, x_label] == label],
                    #             y[labels_final[y_label, x_label] == label],
                    #         )
                    #         xtest = xtest[indexes][:n_nearest]
                    #         ytest = ytest[indexes][:n_nearest]
                    #         for xi, yi in zip(xtest, ytest):
                    #             circle = plt.Circle(
                    #                 (xi, yi),
                    #                 radius_pixels,
                    #                 color="b",
                    #                 fill=False,
                    #             )
                    #             ax.add_artist(circle)
                    #         plt.show()
                depth = calculate_depth(
                    neighbor_values,
                    sigma_level,
                    zero_point,
                    min_number_of_values=min_number_of_values,
                )

            # print(label, depth, depth_diagnostic)
            cat_labels.append(label)
            depths.append(depth)
            diagnostic.append(depth_diagnostic)
            count += 1

        return (
            np.array(depths),
            np.array(diagnostic),
            np.array(cat_labels),
            labels_final,
        )


@jit(nopython=True)
def numba_distances(x=np.array([]), y=np.array([]), x_coords=1, y_coords=1):
    # distances = np.sqrt((x_coords[:, None] - x)**2 + (y_coords[:, None]- y)**2)
    # return distances
    distances = np.zeros_like(x)
    for i in range(len(x)):
        distances[i] = np.sqrt((x[i] - x_coords) ** 2 + (y[i] - y_coords) ** 2)
    return distances


@jit(nopython=True)
def calculate_depth(
    values, sigma_level=5, zero_point=28.0865, min_number_of_values=100
):
    if len(values) < min_number_of_values:
        return np.nan
    median = np.nanmedian(values)
    abs_deviation = np.abs(values - median)
    nmad = 1.4826 * np.nanmedian(abs_deviation) * sigma_level
    if nmad > 0.0:
        depth_sigma = -2.5 * np.log10(nmad) + zero_point
    else:
        depth_sigma = np.nan
    return depth_sigma


def make_ds9_region_file(
    coordinates,
    radius,
    filename,
    coordinate_type="sky",
    convert=True,
    wcs=None,
    pixel_scale=0.03,
):
    """
    coordinates: list of tuples - (x, y) coordinates
    radius: float - the radius of the circles in units of sky or pixels
    filename: str - the name of the file to write the regions to
    coordinate_type: str - 'sky' or 'pixel'
    convert: bool - whether to convert the coordinates to the other coordinate type
    wcs = WCS object - the wcs object to use for the conversion if convert is True
    """
    # If coordinate shape is (2, n) then we have to transpose it
    if np.shape(coordinates)[-1] == 2:
        coordinates = np.array(coordinates).T
    print(f"empty aperture coordinates = {coordinates}")
    x, y = np.array(coordinates)

    if coordinate_type == "sky":
        coord_type = "fk5"
        radius_unit = '"'
        if convert:
            x, y = wcs.all_pix2world(x, y, 0)
            radius = radius * pixel_scale

    elif coordinate_type == "pixel":
        coord_type = "image"
        radius_unit = ""
        if convert:
            x, y = wcs.all_world2pix(x, y, 0)
            radius = radius / pixel_scale

    with open(filename, "w") as f:
        f.write(
            f'# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n{coord_type}\n'
        )
        for xi, yi in zip(x, y):
            f.write(f"circle({xi},{yi},{radius:.5f}{radius_unit})\n")
    funcs.change_file_permissions(filename)


def cluster_wht_map(
    wht_map, num_regions="auto", bin_factor=1, min_size=10000, plot=False
):
    "Works best for 2 regions, but can be used for more than 2 regions - may need additional smoothing and cleaning"
    # Read the image and associated weight map

    # adjust min_size to be in terms of the bin_factor
    min_size = min_size // bin_factor**2
    if isinstance(wht_map, str):
        #
        weight_map = fits.open(wht_map)
        # Check if we have multiple extensions
        if len(weight_map) > 1:
            weight_map = weight_map["WHT"].data
        else:
            weight_map = weight_map[0].data

    elif isinstance(wht_map, np.ndarray):
        weight_map = wht_map

    # Remove NANs
    weight_map[np.isnan(weight_map)] = 0
    percentiles = np.nanpercentile(weight_map, [5, 95])

    weight_map_clipped = np.clip(weight_map, percentiles[0], percentiles[1])
    weight_map_transformed = (
        (weight_map_clipped - percentiles[0])
        / (percentiles[1] - percentiles[0])
        * 255
    )

    weight_map_smoothed = cv2.resize(
        weight_map_transformed,
        (weight_map.shape[1] // bin_factor, weight_map.shape[0] // bin_factor),
        interpolation=cv2.INTER_LINEAR,
    )
    # Renormalize

    # weight_map_smoothed = cv2.normalize(weight_map_smoothed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    percentiles = np.nanpercentile(weight_map_smoothed, [5, 95])
    weight_map_clipped = np.clip(
        weight_map_smoothed, percentiles[0], percentiles[1]
    )
    weight_map_transformed = (
        (weight_map_clipped - percentiles[0])
        / (percentiles[1] - percentiles[0])
        * 255
    )

    weight_map_transformed[np.isnan(weight_map_transformed)] = 0
    labels_filled = []
    iterations = 0
    if num_regions == "auto":
        num_regions_list = [1, 2, 3, 4]
    else:
        num_regions_list = [num_regions]
    sse = []
    if len(num_regions_list) > 1:
        for num_regions in num_regions_list:
            # while len(np.unique(labels_filled)) != num_regions:
            # suming you want to segment into 2 regions (deep and non-deep)
            print(num_regions)
            kmeans = KMeans(n_clusters=num_regions, n_init=5)

            kmeans.fit(weight_map_transformed.flatten().reshape(-1, 1))

            sse.append(kmeans.inertia_)

        if all(val == 0.0 for val in np.unique(sse)):
            print("KMeans failed to find regions. No regions used.")
            return np.zeros_like(weight_map), weight_map_smoothed

        from kneed import KneeLocator

        kneedle = KneeLocator(
            num_regions_list, sse, curve="convex", direction="decreasing"
        )
        num_regions = kneedle.elbow
        print(f"Detected {num_regions} regions as best.")

        # if plot:
        #     plt.plot(num_regions_list, sse)
        #     plt.xlabel("Number of Regions")
        #     plt.ylabel("SSE")
        #     plt.axvline(num_regions, color="red", linestyle="--")
        #     plt.show()
        #     plt.close()
    # Find best of doing it 15x
    kmeans = KMeans(n_clusters=num_regions, n_init=15)
    kmeans.fit(weight_map_transformed.flatten().reshape(-1, 1))

    labels = kmeans.labels_.reshape(weight_map_transformed.shape[:2])

    if num_regions == 2:
        # Closing and opening to remove light and dark spots
        labels_filled = morphology.binary_closing(labels, morphology.disk(5))
        labels_filled = morphology.binary_opening(
            labels_filled, morphology.disk(5)
        )

    else:
        # Do this when you have more than 2 regions - doesn't work quite as well at the edges
        labels_filled = morphology.area_closing(
            labels, area_threshold=min_size
        )
        labels_filled = morphology.area_opening(
            labels_filled, area_threshold=min_size
        )

    # Remove remaining holes
    possible_labels = np.unique(labels_filled)
    for label in possible_labels:
        region = labels_filled == label
        region_cleaned = morphology.remove_small_holes(
            region, area_threshold=min_size
        )
        labels_filled = np.where(region_cleaned, label, labels_filled)

    # Check if both labels are present
    possible_labels = np.unique(labels_filled)
    if len(possible_labels) != num_regions:
        print("One of the Kmeans labelled regions didn't survive cleaning.")
        num_regions = len(possible_labels)

    # Check if one of regions is background (i.e very close to zero)
    zero_levels = [
        np.count_nonzero(weight_map_smoothed[labels_filled == label] < 10)
        / np.count_nonzero(labels_filled == label)
        for label in possible_labels
    ]
    print("Zero levels:", zero_levels)
    possible_background_label = np.argmax(zero_levels)
    background_label = possible_labels[possible_background_label]
    background_frac = zero_levels[possible_background_label]
    if background_frac > 0.80:
        print("Label", int(background_label), "is background")
        if num_regions == 2:
            print(
                "No other regions detected, so no need to break depths into regions."
            )
            labels_filled = np.zeros_like(labels_filled)

    # plt.imshow(weight_map_smoothed, cmap='Greys', origin='lower', interpolation='None')
    # plt.imshow(labels_filled, cmap='viridis', origin='lower', interpolation='None', alpha=0.7)
    # If bin_factor is greater than 1, enlarge the labels_filled to the original size
    if bin_factor > 1:
        labels_filled = cv2.resize(
            labels_filled.astype(np.uint8),
            (weight_map.shape[1], weight_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    show_labels = False
    if show_labels:
        plt.imshow(
            labels_filled, cmap="viridis", origin="lower", interpolation="None"
        )
        plt.show()

    return labels_filled, weight_map_transformed


def get_depth_dir(
    self: Union[Type[Band_Data_Base], Data], aper_diam: u.Quantity, mode: str
) -> str:
    if self.__class__.__name__ in ["Band_Data", "Stacked_Band_Data"]:
        instr_name = self.instr_name
    elif self.__class__.__name__ == "Data":
        instr_name = self.filterset.instrument_name
    else:
        raise ValueError(f"Class {self.__class__.__name__}")
    depth_dir = (
        f"{config['Depths']['DEPTH_DIR']}/"
        + f"{instr_name}/{self.version}/{self.survey}/"
        + f"{format(aper_diam.value, '.2f')}as/{mode}"
    )
    return depth_dir


def get_grid_depth_path(
    self: Type[Band_Data_Base], aper_diam: u.Quantity, mode: str
) -> str:
    depth_dir = get_depth_dir(self, aper_diam, mode)
    depth_path = f"{depth_dir}/{self.filt_name}.h5"
    funcs.make_dirs(depth_path)
    return depth_path


def calc_band_depth(params: Tuple[Any]) -> NoReturn:
    # unpack input parameters
    (
        self,
        aper_diam,
        mode,
        scatter_size,
        distance_to_mask,
        region_radius_used_pix,
        n_nearest,
        coord_type,
        split_depth_min_size,
        split_depths_factor,
        step_size,
        n_split,
        overwrite,
    ) = params

    grid_depth_path = get_grid_depth_path(self, aper_diam, mode)
    if not Path(grid_depth_path).is_file() or overwrite:
        # load the image/segmentation/mask data for the specific band
        im_data, im_header, seg_data, seg_header, mask = self.load_data(
            incl_mask=True
        )
        combined_mask = self.combine_seg_data_and_mask(
            seg_data=seg_data, mask=mask
        )
        wcs = self.load_wcs()
        radius_pix = (aper_diam / (2.0 * self.pix_scale)).value

        # Load wht data if it has the correct type
        wht_data = self.load_wht()
        if n_split is None:
            if wht_data is None:
                n_split = 1
            else:
                n_split = "auto"
        else:
            assert isinstance(n_split, int) or n_split == "auto"

        # Place apertures in empty regions in the image
        xy = make_grid(
            im_data,
            combined_mask,
            radius=(aper_diam / 2.0).value,
            scatter_size=scatter_size,
            distance_to_mask=distance_to_mask,
            plot=False,
        )

        # Make ds9 region file of apertures for compatability and debugging
        region_path = (
            f"{get_depth_dir(self, aper_diam, mode)}/"
            + f"{self.survey}_{self.version}_{self.filt_name}.reg"
        )
        make_ds9_region_file(
            xy,
            radius_pix,
            region_path,
            coordinate_type="pixel",
            convert=False,
            wcs=wcs,
            pixel_scale=self.pix_scale.value,
        )

        # Get fluxes in regions
        fluxes = do_photometry(im_data, xy, radius_pix)
        cat = Table.read(self.forced_phot_path)
        depths, diagnostic, depth_labels, final_labels = calc_depths(
            xy,
            fluxes,
            im_data,
            combined_mask,
            region_radius_used_pix=region_radius_used_pix,
            step_size=step_size,
            catalogue=cat,
            wcs=wcs,
            coord_type=coord_type,
            mode=mode,
            n_nearest=n_nearest,
            zero_point=self.ZP,
            n_split=n_split,
            split_depth_min_size=split_depth_min_size,
            split_depths_factor=split_depths_factor,
            wht_data=wht_data,
        )

        # calculate the depths for plotting purposes
        nmad_grid, num_grid, labels_grid, final_labels = calc_depths(
            xy,
            fluxes,
            im_data,
            combined_mask,
            region_radius_used_pix=region_radius_used_pix,
            step_size=step_size,
            wcs=wcs,
            coord_type=coord_type,
            mode=mode,
            n_nearest=n_nearest,
            zero_point=self.ZP,
            n_split=n_split,
            split_depth_min_size=split_depth_min_size,
            split_depths_factor=split_depths_factor,
            wht_data=wht_data,
            provide_labels=final_labels,
        )

        # write to .h5
        hf_save_names = self.get_depth_h5_labels()
        hf_save_data = [
            mode,
            aper_diam,
            scatter_size,
            distance_to_mask,
            region_radius_used_pix,
            n_nearest,
            split_depth_min_size,
            split_depths_factor,
            step_size,
            depths,
            diagnostic,
            depth_labels,
            final_labels,
            nmad_grid,
            num_grid,
            labels_grid,
        ]
        hf = h5py.File(grid_depth_path, "w")
        for name_i, data_i in zip(hf_save_names, hf_save_data):
            hf.create_dataset(name_i, data=data_i)
        hf.close()

    # if plot:
    #     self.plot_depth(band, cat_creator, mode, aper_diam, show=False)


def get_depth_tab_path(
    self: Type[Band_Data_Base], aper_diam: u.Quantity, mode: str
):
    raise (NotImplementedError)
    depth_dir = get_depth_dir(self, aper_diam, mode)
    depth_tab_path = f"{depth_dir}/{self.survey}_depths.ecsv"
    funcs.make_dirs(depth_tab_path)
    return depth_tab_path


def make_depth_tab(self: Data, mode: str):
    # create .ecsv holding all depths for an instrument if not already written
    depth_tab_path = get_depth_tab_path()
    if not Path(depth_tab_path).is_file():
        depths_tab = None
        calc_filt_aper_diam_arr = [
            (band_data.filt, aper_diam)
            for band_data in self
            for aper_diam in self.aper_diams
        ]
        calc_filt_aper_diam_arr.extend(
            [
                (self.forced_phot_band.filt, aper_diam)
                for aper_diam in self.aper_diams
            ]
        )
    else:
        depths_tab = Table.read(depth_tab_path)
        filt_aper_diams = [
            (filt_name, aper_diam)
            for filt_name, aper_diam in zip(
                depths_tab["filter"], depths_tab["aper_diam"]
            )
        ]
        calc_filt_aper_diam_arr = [
            (band_data.filt, aper_diam)
            for band_data in self
            for aper_diam in aper_diams
            if (band_data.filt_name, f"{format(aper_diam.value, '.2f')}as")
            not in filt_aper_diams
        ]
        calc_filt_aper_diam_arr.extend(
            [
                (self.forced_phot_band.filt, aper_diam)
                for aper_diam in aper_diams
                if (
                    self.forced_phot_band.filt_name,
                    f"{format(aper_diam.value, '.2f')}as",
                )
                not in filt_aper_diams
            ]
        )

    if len(calc_filt_aper_diam_arr) > 0:
        filters = []
        instruments = []
        aper_diams = []
        reg_labels = []
        median_depths = []
        mean_depths = []
        for filt_aper_diam in calc_filt_aper_diam_arr:
            filt = filt_aper_diam[0]
            aper_diam = filt_aper_diam[1]
            med_band_depths, mean_band_depths = get_depths_from_h5(
                filt, aper_diam, mode
            )
            for reg_label in med_band_depths.keys():
                filters.extend([filt.filt_name])
                instruments.extend([filt.instrument.__class__.__name__])
                aper_diams.extend([f"{format(aper_diam.value, '.2f')}as"])
                reg_labels.extend([reg_label])
                median_depths.extend([med_band_depths[reg_label]])
                mean_depths.extend([mean_band_depths[reg_label]])
        new_tab = Table(
            {
                "filter": filters,
                "instrument": instruments,
                "aper_diam": aper_diams,
                "region": reg_labels,
                "median_depth": median_depths,
                "mean_depth": mean_depths,
            },
            dtype=[str, str, str, str, float, float],
        )
        if depths_tab is None:
            tab = new_tab
        else:
            tab = vstack([depths_tab, new_tab])
        if os.access(depth_tab_path, os.W_OK):
            tab.write(depth_tab_path, overwrite=True)


def get_depths_from_h5(
    self: Type[Band_Data_Base], aper_diam: u.Quantity, mode: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    # open .h5
    hf = h5py.File(get_grid_depth_path(self, aper_diam, mode), "r")
    cat_depths = np.array(hf.get("depths"))
    depth_labels = np.array(hf.get("depth_labels"))
    med_reg_band_depths = {
        **{
            str(int(depth_label)): np.nanmedian(
                [
                    depth
                    for depth, label in zip(cat_depths, depth_labels)
                    if label == depth_label
                ]
            )
            for depth_label in np.unique(depth_labels)
            if not np.isnan(depth_label)
        },
        **{"all": np.nanmedian(cat_depths)},
    }
    mean_reg_band_depths = {
        **{
            str(int(depth_label)): np.nanmean(
                [
                    depth
                    for depth, label in zip(cat_depths, depth_labels)
                    if label == depth_label
                ]
            )
            for depth_label in np.unique(depth_labels)
            if not np.isnan(depth_label)
        },
        **{"all": np.nanmean(cat_depths)},
    }
    hf.close()
    return med_reg_band_depths, mean_reg_band_depths


def plot_area_depth(
    self,
    cat_creator,
    mode,
    aper_diam,
    show=False,
    use_area_per_band=True,
    save=True,
    return_array=False,
):
    if cat_creator is None:
        galfind_logger.warning(
            "Could not plot depths as cat_creator == None in Data.plot_area_depth()"
        )
    else:
        self.load_depth_dirs(aper_diam, mode)
        area_tab = self.calc_unmasked_area(
            masking_instrument_or_band_name=self.forced_phot_band,
            forced_phot_band=self.forced_phot_band,
        )
        overwrite = config["Depths"].getboolean("OVERWRITE_DEPTH_PLOTS")
        save_path = f"{self.depth_dirs[aper_diam][mode][self.forced_phot_band]}/depth_areas.png"  # not entirely general -> need to improve self.depth_dirs

        if not Path(save_path).is_file() or overwrite:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            # ax.set_title(f"{self.survey} {self.version} {aper_diam}")
            ax.set_xlabel("Area (arcmin$^{2}$)")
            ax.set_ylabel(r"5$\sigma$ Depth (AB mag)")
            area_row = area_tab[
                area_tab["masking_instrument_band"] == self.forced_phot_band
            ]
            if len(area_row) > 1:
                galfind_logger.warning(
                    f"More than one row found in area_tab for {self.forced_phot_band}! Using the first row."
                )
                area_row = area_row[0]
            area_master = area_row["unmasked_area_total"]
            if type(area_master) == u.Quantity:
                area_master = area_master.value
            area_master = float(area_master)

            bands = self.instrument.band_names.tolist()
            if self.forced_phot_band not in bands:
                bands.append(self.forced_phot_band)
            # cmap = plt.cm.get_cmap("nipy_spectral")
            cmap = plt.cm.get_cmap("RdYlBu_r")
            colors = cmap(np.linspace(0, 1, len(bands)))
            # colors = plt.cm.viridis(np.linspace(0, 1, len(bands)))
            data = {}
            for pos, band in enumerate(bands):
                h5_path = f"{self.depth_dirs[aper_diam][mode][band]}/{band}.h5"

                if overwrite:
                    galfind_logger.info(
                        "OVERWRITE_DEPTH_PLOTS = YES, re-doing depth plots."
                    )

                if not Path(h5_path).is_file():
                    raise (
                        Exception(
                            f"Must first run depths for {self.survey} {self.version} {band} {mode} {aper_diam} before plotting!"
                        )
                    )
                hf = h5py.File(h5_path, "r")
                hf_output = {
                    label: np.array(hf[label])
                    for label in self.get_depth_h5_labels()
                }
                hf.close()
                # Need unmasked area for each band
                if use_area_per_band:
                    area_tab = self.calc_unmasked_area(
                        masking_instrument_or_band_name=band,
                        forced_phot_band=self.forced_phot_band,
                    )
                    area_row = area_tab[
                        area_tab["masking_instrument_band"] == band
                    ]

                area = area_row["unmasked_area_total"].to(u.arcmin**2).value

                total_depths = hf_output["nmad_grid"].flatten()
                total_depths = total_depths[~np.isnan(total_depths)]
                total_depths = total_depths[total_depths != 0]
                total_depths = total_depths[total_depths != np.inf]

                # Round to 0.01 mag and sort
                # total_depths = np.round(total_depths, 2)
                total_depths = np.flip(np.sort(total_depths))

                # Calculate the cumulative distribution scaled to area of band
                n = len(total_depths)
                cum_dist = np.arange(1, n + 1) / n
                cum_dist = cum_dist * area

                # Plot
                ax.plot(
                    cum_dist,
                    total_depths,
                    label=band if "+" not in band else "Detection",
                    color=colors[pos] if "+" not in band else "black",
                    drawstyle="steps-post",
                    linestyle="solid" if "+" not in band else "dashed",
                )
                if return_array:
                    data[band] = [area, total_depths]
                # Set ylim to 2nd / 98th percentile if depth is smaller than this number
                ylim = ax.get_ylim()

                if pos == 0:
                    min_depth = np.percentile(total_depths, 0.5)
                    max_depth = np.percentile(total_depths, 99.5)
                else:
                    min_temp = np.percentile(total_depths, 0.5)
                    max_temp = np.percentile(total_depths, 99.5)
                    if min_temp < min_depth:
                        min_depth = min_temp
                    if max_temp > max_depth:
                        max_depth = max_temp

            ax.set_ylim(max_depth, min_depth)
            # Place legend under plot
            ax.legend(
                frameon=False,
                ncol=4,
                bbox_to_anchor=(0.5, -0.14),
                loc="upper center",
                fontsize=8,
                columnspacing=1,
                handletextpad=0.5,
            )
            # ax.legend(frameon = False, ncol = 2)
            # Add inner ticks
            from matplotlib.ticker import AutoMinorLocator

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            # Make ticks face inwards
            ax.tick_params(direction="in", axis="both", which="both")
            # Set minor ticks to face in

            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_ticks_position("both")

            ax.set_xlim(0, area_master * 1.02)
            # Add hlines at integer depths
            depths = np.arange(20, 35, 1)
            # for depth in depths:
            #    ax.hlines(depth, 0, area_master, color = "black", linestyle = "dotted", alpha = 0.5)
            # Invert y axis
            # ax.invert_yaxis()
            ax.grid(True)
            if save:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            if return_array:
                return data


def plot_depth(
    self, band, cat_creator, mode, aper_diam, show=False
):  # , **kwargs):
    if type(cat_creator) == type(None):
        galfind_logger.warning(
            "Could not plot depths as cat_creator == None in Data.plot_depth()"
        )
    else:
        self.load_depth_dirs(aper_diam, mode)
        save_path = (
            f"{self.depth_dirs[aper_diam][mode][band]}/{band}_depths.png"
        )
        # determine paths and whether to overwrite
        overwrite = config["Depths"].getboolean("OVERWRITE_DEPTH_PLOTS")
        if overwrite:
            galfind_logger.info(
                "OVERWRITE_DEPTH_PLOTS = YES, re-doing depth plots."
            )
        if not Path(save_path).is_file() or overwrite:
            # load depth data
            h5_path = f"{self.depth_dirs[aper_diam][mode][band]}/{band}.h5"
            if not Path(h5_path).is_file():
                raise (
                    Exception(
                        f"Must first run depths for {self.survey} {self.version} {band} {mode} {aper_diam} before plotting!"
                    )
                )
            hf = h5py.File(h5_path, "r")
            hf_output = {
                label: np.array(hf[label])
                for label in self.get_depth_h5_labels()
            }
            hf.close()
            # load image and wcs
            im_data, im_header = self.load_im(band)
            wcs = WCS(im_header)
            # make combined mask
            combined_mask = self.combine_seg_data_and_mask(band)
            # load catalogue to calculate x/y image coordinates
            cat = Table.read(self.sex_cat_master_path)
            cat_x, cat_y = wcs.world_to_pixel(
                SkyCoord(
                    cat[cat_creator.ra_dec_labels["RA"]],
                    cat[cat_creator.ra_dec_labels["DEC"]],
                )
            )

            show_depths(
                hf_output["nmad_grid"],
                hf_output["num_grid"],
                hf_output["step_size"],
                hf_output["region_radius_used_pix"],
                hf_output["labels_grid"],
                hf_output["depth_labels"],
                hf_output["depths"],
                hf_output["diagnostic"],
                cat_x,
                cat_y,
                combined_mask,
                hf_output["final_labels"],
                suptitle=f"{self.survey} {self.version} {band} Depths",
                save_path=save_path,
                show=show,
            )


def get_depth_args(params: List[Tuple[Any, ...]]) -> Dict[str, Any]:
    return {
        "mode": params[2],
        "scatter_size": params[3],
        "distance_to_mask": params[4],
        "region_radius_used_pix": params[5],
        "n_nearest": params[6],
        "coord_type": params[7],
        "split_depth_min_size": params[8],
        "split_depths_factor": params[9],
        "step_size": params[10],
        "n_split": params[11],
    }


def get_depth_h5_labels():
    return [
        "mode",
        "aper_diam",
        "scatter_size",
        "distance_to_mask",
        "region_radius_used_pix",
        "n_nearest",
        "split_depth_min_size",
        "split_depths_factor",
        "step_size",
        "depths",
        "diagnostic",
        "depth_labels",
        "final_labels",
        "nmad_grid",
        "num_grid",
        "labels_grid",
    ]

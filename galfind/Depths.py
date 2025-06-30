# Depths.py
from __future__ import annotations

# import automask as am
import astropy.visualization as vis
import cv2 as cv2
import matplotlib as mpl
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
import logging
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numba import jit
from photutils.aperture import CircularAperture
from scipy.stats import gaussian_kde
from skimage import morphology
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Optional, Union, Tuple, Dict, List, Any, NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Band_Data_Base, Data

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

def make_grid_force(data, mask, radius, scatter_size, pixel_scale=0.03, plot=False, ax=None, distance_to_mask=50, n_retry_box=5, grid_offset_times=4):
    radius_pixels = radius / pixel_scale
    scatter_size_pixels = scatter_size / pixel_scale
    
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance_to_mask, distance_to_mask))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(bool)
    
    placed_apertures_mask = np.zeros_like(mask)
    
    shifts = [(0, 0), (0, scatter_size_pixels/2), (scatter_size_pixels/2, 0), (scatter_size_pixels/2, scatter_size_pixels/2)]
    shift = lambda i: shifts[i % 4] if i < 4 else (shifts[i % 4][0] * (i // 4), shifts[i % 4][1] * (i // 4))
    
    non_overlapping_xy = []

    for i in range(grid_offset_times):
        x_shift, y_shift = shift(i)
        
        xy = np.mgrid[radius_pixels + scatter_size_pixels+x_shift:data.shape[1]-(radius_pixels + scatter_size_pixels):2*(radius_pixels + scatter_size_pixels),
                     radius_pixels + scatter_size_pixels:data.shape[0]-(radius_pixels + scatter_size_pixels)+y_shift:2*(radius_pixels+scatter_size_pixels)]
        
        xy = xy.reshape(2, -1).T
        scatter = np.random.uniform(low=-scatter_size_pixels, high=scatter_size_pixels, size=(xy.shape[0], 2))
        xy_scatter = xy + scatter
        
        for pos, (x, y) in tqdm(enumerate(xy_scatter), disable=galfind_logger.getEffectiveLevel() > logging.INFO):
            count = 0
            done = False
            
            while count < n_retry_box and not done:
                y_min = int(np.floor(y-radius_pixels))
                y_max = int(np.ceil(y+radius_pixels))
                x_min = int(np.floor(x-radius_pixels))
                x_max = int(np.ceil(x+radius_pixels))
                
                if y_min < 0 or x_min < 0 or y_max > mask.shape[0] or x_max > mask.shape[1]:
                    break
                    
                mask_cutout = mask[y_min:y_max, x_min:x_max]
                placed_cutout = placed_apertures_mask[y_min:y_max, x_min:x_max]
                
                y_center_internal = mask_cutout.shape[0] // 2
                x_center_internal = mask_cutout.shape[1] // 2
                
                delta_shape = mask_cutout.shape[0] - mask_cutout.shape[1]
                if delta_shape != 0 and abs(delta_shape) >= scatter_size_pixels:
                    break
                    
                y_temp, x_temp = np.ogrid[-y_center_internal:mask_cutout.shape[0]-y_center_internal, 
                                        -x_center_internal:mask_cutout.shape[1]-x_center_internal]
                inside_pixels = x_temp**2 + y_temp**2 <= radius_pixels**2
                
                if not np.any(mask_cutout[inside_pixels]) and not np.any(placed_cutout[inside_pixels]):
                    non_overlapping_xy.append((x, y))
                    
                    # Create circle mask with exact dimensions
                    size = int(2 * radius_pixels + 1)
                    y_grid, x_grid = np.ogrid[-radius_pixels:radius_pixels+1, -radius_pixels:radius_pixels+1]
                    circle_mask = x_grid**2 + y_grid**2 <= radius_pixels**2
                    
                    # Calculate exact region to update
                    y_center, x_center = int(y), int(x)
                    y_start = max(0, y_center - int(radius_pixels))
                    y_end = min(placed_apertures_mask.shape[0], y_center + int(radius_pixels) + 1)
                    x_start = max(0, x_center - int(radius_pixels))
                    x_end = min(placed_apertures_mask.shape[1], x_center + int(radius_pixels) + 1)
                    
                    # Extract the exact portion of circle_mask needed
                    mask_y_start = int(radius_pixels - (y_center - y_start))
                    mask_y_end = int(radius_pixels + (y_end - y_center))
                    mask_x_start = int(radius_pixels - (x_center - x_start))
                    mask_x_end = int(radius_pixels + (x_end - x_center))
                    
                    placed_apertures_mask[y_start:y_end, x_start:x_end] |= \
                        circle_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                    done = True
                else:
                    x, y = xy_scatter[pos] + np.random.uniform(low=-scatter_size_pixels, high=scatter_size_pixels, size=2)
                    count += 1
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
            
        stretch = vis.CompositeStretch(vis.LogStretch(), vis.ContrastBiasStretch(contrast=30, bias=0.08))    
        norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)
        
        ax.imshow(data, cmap='Greys', origin='lower', interpolation='None', norm=norm)
        ax.imshow(mask, cmap='Reds', alpha=0.5, origin='lower', interpolation='None')
        ax.imshow(placed_apertures_mask, cmap='Blues', alpha=0.5, origin='lower', interpolation='None')
        
        for x, y in non_overlapping_xy:
            circle = plt.Circle((x, y), radius_pixels, color='r', fill=False)
            ax.add_artist(circle)   
            
        plt.show()
        
    possible_pos = ((data.shape[0] * data.shape[1]) - np.sum(mask)) / (np.pi * radius_pixels ** 2)
    placing_efficiency = len(non_overlapping_xy) / possible_pos
    galfind_logger.info(f"Placing efficiency = {100 * placing_efficiency:.2f}%")
    
    return non_overlapping_xy, placing_efficiency


def make_grid(
    data,
    mask,
    radius,
    scatter_size,
    pixel_scale=0.03,
    plot=False,
    ax=None,
    distance_to_mask=30,
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
    #print(xy.shape)

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
            non_overlapping_xy.extend(
                [(x, y)]
            )  # Add non-overlapping coordinates to the list
    #print("Number of non-overlapping apertures:", len(non_overlapping_xy))

    possible_pos = \
        ((data.shape[0] * data.shape[1]) - np.sum(mask)) / \
        (np.pi * radius_pixels ** 2)
    placing_efficiency = len(non_overlapping_xy) / possible_pos

    plot=True
    # Plot the circles using matplotlib
    if plot:
        if ax == None:
            fig, ax = plt.subplots()

        stretch = vis.CompositeStretch(vis.LogStretch(), vis.ContrastBiasStretch(contrast=30, bias=0.08))    
        norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)

        ax.imshow(data, cmap='Greys', origin='lower', interpolation='None',
        norm=norm)

        # imshow mask
        ax.imshow(mask, cmap='Reds', alpha=0.5, origin='lower', interpolation='None')

        for x, y in non_overlapping_xy:
            circle = plt.Circle((x, y), radius_pixels, color='r', fill=False)
            ax.add_artist(circle)   

        plt.show()


    #print(scatter_size_pixels, distance_to_mask)
    galfind_logger.info(f"Placing efficiency = {100 * placing_efficiency:.2f}%")
    return non_overlapping_xy, placing_efficiency


def calc_depths(
    coordinates,
    fluxes,
    img_data,
    mask=None,
    catalogue=None,
    mode="n_nearest",
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
        for i in tqdm(range(0, grid_size[0], step_size), disable = galfind_logger.getEffectiveLevel() > logging.INFO):
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
                        if len(distances_i) < n_nearest:
                            nearest_indices = np.arange(len(distances_i))
                            min_number_of_values = len(distances_i)
                        else:
                            nearest_indices = np.argpartition(distances_i, n_nearest)[
                                :n_nearest
                            ]
                            min_number_of_values = n_nearest
                        neighbor_values = fluxes_i[nearest_indices]

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
        for i, j in tqdm(zip(cat_x, cat_y), total=len(cat_x), disable=galfind_logger.getEffectiveLevel() > logging.INFO):
            # Check if the coordinate is outside the image or in the mask
            if i > x_max or i < 0 or j > y_max or j < 0:
                depth = np.nan
                num_of_apers = np.nan
                label = np.nan
                depth_diagnostic = np.nan
            else:

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
                    if len(distances_i) < n_nearest:
                        nearest_indices = np.arange(len(distances_i))
                        min_number_of_values = len(distances_i)
                    # print(labels_final.dtype, y_label.dtype, x_label.dtype, fluxes.dtype, n_nearest.dtype, label.dtype)
                    # neighbor_values, depth_diagnostic = numba_n_nearest_filter(fluxes_i, distances_i, n_nearest)
                    else:
                        nearest_indices = np.argpartition(distances_i, n_nearest)[
                            :n_nearest
                        ]
                        min_number_of_values = n_nearest
                    neighbor_values = fluxes_i[nearest_indices]

                    # Depth diagnostic is distance to n_nearest
                    depth_diagnostic = min_number_of_values  # distances_i[np.argsort(distances_i[nearest_indices])][-1]

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
        weight_map = fits.open(wht_map, ignore_missing_simple = True)
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

        if num_regions is None:
            num_regions = 1

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
        n_retry_box,
        grid_offset_times,
        overwrite,
        master_cat_path,
    ) = params

    grid_depth_path = get_grid_depth_path(self, aper_diam, mode)
    if not Path(grid_depth_path).is_file() or overwrite:
        # load the image/segmentation/mask data for the specific band
        im_data = self.load_im()[0]
        combined_mask = self._combine_seg_data_and_mask()
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
        xy, placing_efficiency = make_grid_force(
            im_data,
            combined_mask,
            radius=(aper_diam / 2.0).value,
            scatter_size=scatter_size,
            distance_to_mask=distance_to_mask,
            pixel_scale=self.pix_scale.value,
            n_retry_box=n_retry_box,
            grid_offset_times=grid_offset_times,
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
        if master_cat_path is None:
            cat = Table.read(self.forced_phot_path)
        else:
            cat = Table.read(master_cat_path)
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
        hf_save_names = get_depth_h5_labels()
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
            placing_efficiency,
            n_retry_box,
            grid_offset_times,
        ]
        assert len(hf_save_names) == len(hf_save_data)

        hf = h5py.File(grid_depth_path, "w")
        for name_i, data_i in zip(hf_save_names, hf_save_data):
            hf.create_dataset(
                name_i,
                data = data_i,
                compression = "gzip" if isinstance(data_i, np.ndarray) and \
                    not isinstance(data_i, tuple([u.Quantity, u.Magnitude, u.Dex])) \
                    else None
                )
        hf.close()


def get_depth_tab_path(self: Data) -> str:
    depth_dir = get_depth_tab_dir(self)
    depth_tab_path = f"{depth_dir}/{self.survey}_depths.ecsv"
    funcs.make_dirs(depth_tab_path)
    return depth_tab_path


def get_depth_tab_dir(self: Data) -> str:
    return (
        f"{config['Depths']['DEPTH_DIR']}/Depth_tables/"
        + f"{self.version}/{self.survey}"
    )


def make_depth_tab(self: Data) -> NoReturn:
    # create .ecsv holding all depths for an instrument if not already written
    depth_tab_path = get_depth_tab_path(self)
    if not Path(depth_tab_path).is_file():
        depths_tab = None
        calc_params_arr = [
            (band_data, aper_diam, 
            band_data.depth_args[aper_diam]["mode"])
            for band_data in self
            for aper_diam in band_data.aper_diams
        ]
        if self.forced_phot_band not in self:
            calc_params_arr.extend(
                [
                    (self.forced_phot_band, aper_diam, 
                    self.forced_phot_band.depth_args[aper_diam]["mode"])
                    for aper_diam in self.forced_phot_band.aper_diams
                ]
            )
    else:
        depths_tab = Table.read(depth_tab_path)
        filt_aper_diams = [
            (filt_name, aper_diam, mode)
            for filt_name, aper_diam, mode in zip(
                depths_tab["filter"], 
                depths_tab["aper_diam"],
                depths_tab["mode"]
            )
        ]
        calc_params_arr = [
            (band_data, aper_diam, band_data.depth_args[aper_diam]["mode"])
            for band_data in self
            for aper_diam in band_data.aper_diams
            if (band_data.filt_name, f"{format(aper_diam.value, '.2f')}as", 
            band_data.depth_args[aper_diam]["mode"]) not in filt_aper_diams
        ]
        if self.forced_phot_band not in self:
            calc_params_arr.extend(
                [
                    (self.forced_phot_band, aper_diam, 
                    self.forced_phot_band.depth_args[aper_diam]["mode"])
                    for aper_diam in self.forced_phot_band.aper_diams
                    if (
                        self.forced_phot_band.filt_name,
                        f"{format(aper_diam.value, '.2f')}as",
                        self.forced_phot_band.depth_args[aper_diam]["mode"]
                    )
                    not in filt_aper_diams
                ]
            )

    if len(calc_params_arr) > 0:
        filters = []
        instruments = []
        aper_diams = []
        modes = []
        reg_labels = []
        median_depths = []
        mean_depths = []
        for calc_params in calc_params_arr:
            band_data = calc_params[0]
            aper_diam = calc_params[1]
            mode = calc_params[2]
            med_band_depths, mean_band_depths = get_depths_from_h5(
                band_data, aper_diam, mode
            )
            for reg_label in med_band_depths.keys():
                filters.extend([band_data.filt_name])
                if band_data.__class__.__name__ == "Band_Data":
                    instruments.extend([band_data.filt.instrument.__class__.__name__])
                else:
                    instruments.extend([band_data.filterset[0].instrument.__class__.__name__])
                aper_diams.extend([f"{format(aper_diam.value, '.2f')}as"])
                modes.extend([mode])
                reg_labels.extend([reg_label])
                median_depths.extend([med_band_depths[reg_label]])
                mean_depths.extend([mean_band_depths[reg_label]])
        new_tab = Table(
            {
                "filter": filters,
                "instrument": instruments,
                "aper_diam": aper_diams,
                "mode": modes,
                "region": reg_labels,
                "median_depth": median_depths,
                "mean_depth": mean_depths,
            },
            dtype=[str, str, str, str, str, float, float],
        )
        if depths_tab is None:
            tab = new_tab
        else:
            tab = vstack([depths_tab, new_tab])
        if os.access(depth_tab_path, os.W_OK) or depths_tab is None:
            tab.write(depth_tab_path, overwrite=True)
            galfind_logger.info(
                f"Depth table written to {depth_tab_path}"
            )
        else:
            galfind_logger.info(
                f"No permissions for {depth_tab_path}"
            )


def get_depths_from_h5(
    self: Type[Band_Data_Base],
    aper_diam: u.Quantity,
    mode: str,
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


def get_depth_plot_path(
    self: Type[Band_Data_Base],
    aper_diam: u.Quantity
) -> str:
    depth_dir = f"{get_depth_dir(self, aper_diam, self.depth_args[aper_diam]['mode'])}/plots"
    depth_plot_path = f"{depth_dir}/{self.filt_name}.png"
    funcs.make_dirs(depth_plot_path)
    return depth_plot_path


def get_area_depth_dir(self: Data) -> str:
    return (
        f"{config['Depths']['DEPTH_DIR']}/Depth_area_plots/"
        + f"{self.version}/{self.survey}"
    )


def plot_area_depth(
    self: Data,
    aper_diam: u.Quantity,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    use_area_per_band=True,
    save: bool = True,
    return_array=False,
    cmap_name: str = "RdYlBu_r",
    overwrite: bool = True
) -> NoReturn:
    # ensure the depths for all bands are calculated in the given aper_diam
    assert all(aper_diam in band_data.depth_args.keys() for band_data in self)
    
    area_tab = None #self.calc_unmasked_area(
    #     masking_instrument_or_band_name=self.forced_phot_band,
    #     forced_phot_band=self.forced_phot_band,
    # )

    if hasattr(self, "forced_phot_band"):
        if self.forced_phot_band not in self:
            self.band_data_arr = self.band_data_arr + [self.forced_phot_band]
        else:
            self_band_data = self.band_data_arr
    else:
        self_band_data = self.band_data_arr

    save_path = f"{get_area_depth_dir(self)}/{format(aper_diam.value, '.2f')}as_area_depth.png"
    funcs.make_dirs(save_path)

    if not Path(save_path).is_file() or overwrite:

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        cmap = plt.cm.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, len(self_band_data)))

        data = {}
        for i, band_data in enumerate(self_band_data):
            area = None
            hf_output = get_hf_output(band_data, aper_diam)

            total_depths = hf_output["nmad_grid"].flatten()
            total_depths = total_depths[~np.isnan(total_depths)]
            total_depths = total_depths[total_depths != 0]
            total_depths = total_depths[total_depths != np.inf]
            total_depths = np.flip(np.sort(total_depths))
            # Calculate the cumulative distribution scaled to area of band
            cum_dist = np.arange(1, len(total_depths) + 1) * area / len(total_depths)

            # Plot
            ax.plot(
                cum_dist,
                total_depths,
                label=band_data.filt_name,
                color=colors[i] if band_data.__class__.__name__ == "Band_Data" else "black",
                drawstyle="steps-post",
                linestyle="solid" if band_data.__class__.__name__ == "Band_Data" else "dashed",
            )

            if i == 0:
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
        
        # ax.set_xlim(0, area_master * 1.02)
        # Add hlines at integer depths
        depths = np.arange(20, 35, 1)
        # for depth in depths:
        #    ax.hlines(depth, 0, area_master, color = "black", linestyle = "dotted", alpha = 0.5)
        ax.grid(True)
        if save:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

def get_hf_output(
    self: Type[Band_Data_Base], 
    aper_diam: u.Quantity,
) -> Dict[str, Any]:
    # open .h5
    h5_path = get_grid_depth_path(self, aper_diam, self.depth_args[aper_diam]["mode"])
    assert Path(h5_path).is_file(), \
        galfind_logger.critical(
            f"{h5_path} does not exist!"
        )
    hf = h5py.File(h5_path, "r")
    hf_output = {
        label: np.array(hf[label])
        for label in get_depth_h5_labels()
        if label in hf.keys()
    }
    hf.close()
    hf_output["nmad_grid"][hf_output["nmad_grid"] == 0] = np.nan
    return hf_output

def get_cat_xy(
    self: Type[Band_Data_Base],
    master_cat_path: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    # load catalogue to calculate x/y image coordinates
    if master_cat_path is None:
        cat = Table.read(self.forced_phot_path)
    else:
        cat = Table.read(master_cat_path)
    cat_ra = cat[self.forced_phot_args["ra_label"]]
    if cat_ra.unit is None:
        cat_ra *= self.forced_phot_args["ra_unit"]
    cat_dec = cat[self.forced_phot_args["dec_label"]]
    if cat_dec.unit is None:
        cat_dec *= self.forced_phot_args["dec_unit"]
    wcs = self.load_wcs()
    cat_x, cat_y = wcs.world_to_pixel(SkyCoord(cat_ra, cat_dec))
    return cat_x, cat_y

def plot_depth_diagnostic(
    self: Type[Band_Data_Base],
    aper_diam: u.Quantity,
    save: bool = True, 
    show: bool = False,
    cmap: str = "plasma",
    master_cat_path: Optional[str] = None
) -> NoReturn:
    plt.style.use("default")
    assert hasattr(self, "depth_path"), \
        galfind_logger.critical(
            f"{repr(self)} has no 'depth_path'"
        )
    hf_output = get_hf_output(self, aper_diam)
    cat_x, cat_y = get_cat_xy(self, master_cat_path)
    combined_mask = self._combine_seg_data_and_mask()

    # setup figure and axes appropriately
    fig, axs = plt.subplots(
        3 if hf_output["labels_grid"] is None else 4,
        1 if hf_output["depths"] is None else 2,
        facecolor="white",
        figsize=(16, 16),
        constrained_layout=True,
    )
    axs = axs.flatten()
    fig.suptitle(
        f"{self.survey} {self.version} {self.filt_name} Depths", 
        fontsize="large", 
        fontweight="bold"
    )
    cmap_ = cm.get_cmap(cmap)
    cmap_.set_bad(color="black")

    labels_arr, possible_labels, colours, labels_cmap = _get_labels(hf_output, cmap_name = "Set2")

    _plot_rolling_average(fig, axs[0], hf_output, cmap_)
    _plot_rolling_average_diagnostic(fig, axs[1], hf_output, cmap_)
    _plot_cat_depths(fig, axs[2], hf_output, cmap_, cat_x, cat_y, combined_mask)
    _plot_depth_hist(fig, axs[3], hf_output, labels_arr, possible_labels, colours)
    _plot_cat_diagnostic(fig, axs[4], hf_output, cmap_, cat_x, cat_y, combined_mask)
    if len(possible_labels) > 1:
        _plot_labels(axs[5], hf_output, labels_cmap)

    # Delete all axes with nothing on
    for i, ax in enumerate(axs):
        if (
            len(ax.images) == 0
            and len(ax.collections) == 0
            and len(ax.lines) == 0
            and len(ax.patches) == 0
            and len(ax.artists) == 0
        ):
            try:
                ax.remove()
                galfind_logger.debug(
                    f"Removed empty axis {i} in depth diagnostic plot"
                )
            except:
                pass

    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)

    if save:
        save_path = get_depth_plot_path(self, aper_diam)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        funcs.change_file_permissions(save_path)
        galfind_logger.info(f"Saved depths plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.clf()

def _plot_rolling_average(
    fig: plt.Figure, 
    ax: plt.Axes, 
    hf_output: Dict[str, Any],
    cmap: plt.Colormap
) -> NoReturn:
    # Make vmin and vmax the 1st and 99th percentile of the nmad_grid
    vmin, vmax = (
        np.nanpercentile(hf_output["nmad_grid"], 1),
        np.nanpercentile(hf_output["nmad_grid"], 99),
    )

    mappable = ax.imshow(
        hf_output["nmad_grid"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
    )
    radius = hf_output["region_radius_used_pix"] / hf_output["step_size"]
    circle_x = radius + 0.07 * np.shape(hf_output["nmad_grid"])[1]
    circle_y = radius + 0.07 * np.shape(hf_output["nmad_grid"])[0]

    patch = plt.Circle(
        (circle_x, circle_y),
        radius=radius,
        fill=True,
        facecolor="white",
        lw=2,
        zorder=10,
    )
    ax.add_patch(patch)
    ax.text(
        circle_x,
        circle_y - 1.4 * radius,
        s="Filter Size",
        va="center",
        ha="center",
        color="white",
        fontsize="medium",
        path_effects = [pe.Stroke(linewidth=0.5, foreground='black')]
    )
    # Make colorbar same height as ax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mappable, label=r"5$\sigma$ Depth", cax=cax)
    ax.set_title(r"Rolling Average 5$\sigma$ Depth")

def _plot_rolling_average_diagnostic(
    fig: plt.Figure, 
    ax: plt.Axes, 
    hf_output: Dict[str, Any],
    cmap: plt.Colormap
) -> NoReturn:
    mappable = ax.imshow(hf_output["num_grid"], origin="lower", cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mappable, label=r"Number of Apertures Used", cax=cax)
    ax.set_title("Rolling Average Diagnostic")

def _get_labels(
    hf_output: Dict[str, Any],
    cmap_name: str = "Set2"
) -> Tuple[List[str], List[int], List[plt.Color], plt.Colormap]:
    possible_labels = np.unique(hf_output["labels_grid"])
    av_depths = [
        np.nanmedian(hf_output["nmad_grid"][hf_output["labels_grid"] == label])
        for label in possible_labels
    ]
    # mask nan in av_depths
    av_depths = np.array(av_depths)[~np.isnan(av_depths)]
    num_labels = len(np.unique(hf_output["labels_grid"]))
    labels_cmap = LinearSegmentedColormap.from_list(
        "custom",
        [cm.get_cmap(cmap_name)(i / num_labels) for i in range(num_labels)],
        num_labels,
    )
    if len(av_depths) == 1:
        labels_arr = ["Single Region"]
        possible_labels = [0]
        colours = [cm.get_cmap(cmap_name)(0)]
    elif len(av_depths) == 2:
        labels_arr = ["Shallow", "Deep"]
        colours = [
            #[cm.get_cmap(cmap_name)(i / len(av_depths)) for i in range(len())]
            labels_cmap(possible_labels[0]),
            labels_cmap(possible_labels[1]),
        ]
    else:
        err_message = "Depth plotting fails with more than 2 regions!"
        galfind_logger.critical(err_message)
        raise Exception(err_message)
    return labels_arr, possible_labels, colours, labels_cmap

def _plot_labels(
    ax: plt.Axes,
    hf_output: Dict[str, Any],
    cmap: plt.Colormap,
) -> Tuple[List[str], List[int], List[plt.Color]]:
    ax.imshow(
        hf_output["labels_grid"], cmap=cmap, origin="lower", interpolation="None"
    )
    ax.set_title("Labels")

def _plot_cat_depths(
    fig: plt.Figure,
    ax: plt.Axes,
    hf_output: Dict[str, Any],
    cmap: plt.Colormap,
    cat_x: np.ndarray,
    cat_y: np.ndarray,
    combined_mask: np.ndarray,
) -> NoReturn:
    ax.set_title("Catalogue Depths")
    m = ax.scatter(
        cat_x, cat_y, s=1, zorder=5, c=hf_output["depths"], cmap=cmap, edgecolors=None
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mappable=m, label=r"5$\sigma$ Depth", cax=cax)
    ax.imshow(
        combined_mask,
        cmap="Greens",
        origin="lower",
        interpolation="None",
        alpha=0.3,
        zorder=4,
    )
    ax.imshow(
        hf_output["final_labels"],
        cmap="Reds",
        origin="lower",
        interpolation="None",
        alpha=0.3,
        zorder=4,
    )

def _plot_cat_diagnostic(
    fig: plt.Figure,
    ax: plt.Axes,
    hf_output: Dict[str, Any],
    cmap: plt.Colormap,
    cat_x: np.ndarray,
    cat_y: np.ndarray,
    combined_mask: np.ndarray,
) -> NoReturn:
    ax.set_title("Catalogue Diagnostic")
    m = ax.scatter(
        cat_x,
        cat_y,
        s=1,
        zorder=5,
        c=np.array(hf_output["diagnostic"]),
        cmap=cmap,
        edgecolors=None
    )
    # Make it so axes aren't stetch disproportionately
    ax.set_aspect("equal")
    divider4 = make_axes_locatable(ax)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        mappable=m, label="Distance to 200th Empty Aperture", cax=cax4
    )

def _plot_depth_hist(
    fig: plt.Figure,
    ax: plt.Axes,
    hf_output: Dict[str, Any],
    labels_arr: List[str],
    possible_labels: List[int],
    colours: List[plt.Color],
    annotate: bool = True,
    label_suffix: Optional[str] = None,
    title: Optional[str] = None # default None prints "Depth Histogram"
) -> NoReturn:
    set_labels = [
        hf_output["depths"][hf_output["depth_labels"] == label] for label in possible_labels
    ]
    for set_label, label, colour in zip(set_labels, labels_arr, colours):
        _set_label = set_label[~np.isnan(set_label)]  # remove nans
        if label_suffix is not None:
            label += f" {label_suffix}"
        ax.hist(
            _set_label,
            bins=40,
            range=(np.nanmin(hf_output["depths"]), np.nanmax(hf_output["depths"])),
            label=label,
            color=colour,
            histtype="stepfilled",
            alpha=0.8,
        )
    # Plot line at median depth
    # Fix y range
    ax.set_ylim(ax.get_ylim())
    max = ax.get_ylim()[1]
    for pos, (depth, colour) in enumerate(zip(set_labels, colours)):
        ax.axvline(
            np.nanmedian(depth),
            0,
            max,
            color="black",
            lw=3,
            linestyle="--",
            label="Median" if pos == 0 and annotate else "",
            zorder=10,
        )
        ax.axvline(
            np.nanmean(depth),
            0,
            max,
            color="black",
            lw=3,
            linestyle="-",
            label="Mean" if pos == 0 and annotate else "",
            zorder=10,
        )
        # Label with text
        ax.text(
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
        ax.text(
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
    if annotate:
        ax.set_xlabel(r"5$\sigma$ Depth")
        if title is None:
            title = "Depth Histogram"
        ax.set_yticks([])
        ax.set_title(title)
        ax.legend(frameon=False)


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
        "n_retry_box": params[12],
        "grid_offset_times": params[13],
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
        "placing_efficiency",
        "n_retry_box",
        "grid_offset_times",
    ]

def append_loc_depth_cols(
    self: Data, 
    min_flux_pc_err: Optional[Union[int, float]] = None, 
    overwrite: bool = False
) -> NoReturn:
    # open catalogue
    cat = Table.read(self.phot_cat_path)
    # update catalogue with local depths if not already done so
    if not (f"FLUX_APER_{self[0].filt_name}_aper_corr" in cat.colnames):
        galfind_logger.critical(
            "Must run aperture corrections before appending local depth columns!"
        )
    elif f"loc_depth_{self[0].filt_name}" not in cat.colnames or overwrite:
        assert hasattr(self, "forced_phot_band"), \
            galfind_logger.critical(
                f"{repr(self)} has no 'forced_phot_band'"
            )
        # ensure aperture diameters are the same for all bands
        assert all(all(diam == diam_0 for diam, diam_0 in 
            zip(band_data.aper_diams, self[0].aper_diams)) 
            for band_data in self), galfind_logger.critical(
                f"Aperture diameters are not the same for all bands in {repr(self)}"
            )
        if overwrite:
            # TODO: Delete already existing columns
            raise(Exception())
        aper_diams = self[0].aper_diams.to(u.arcsec).value
        for i, band_data in tqdm(enumerate(self.band_data_arr), 
                total=len(self), desc="Appending local depth columns",
                disable=galfind_logger.getEffectiveLevel() > logging.INFO):
            for j, aper_diam in enumerate(aper_diams):
                aper_diam *= u.arcsec
                h5_path = get_grid_depth_path(
                    band_data, aper_diam, band_data.depth_args[aper_diam]["mode"]
                )
                if Path(h5_path).is_file():
                    # open depth .h5
                    hf = h5py.File(h5_path, "r")
                    depths = np.array(hf["depths"])
                    diagnostics = np.array(hf["diagnostic"])
                    diagnostic_name_ = (
                        f"d_{int(np.array(hf['n_nearest']))}"
                        if band_data.depth_args[aper_diam]["mode"] == "n_nearest"
                        else f"n_aper_{float(np.array(hf['region_radius_used_pix'])):.1f}"
                        if band_data.depth_args[aper_diam]["mode"] == "rolling"
                        else None
                    )
                    # make sure the same depth setup has been run in each band
                    if i == 0 and j == 0:
                        diagnostic_name = diagnostic_name_
                    assert diagnostic_name_ == diagnostic_name
                    hf.close()
                else:
                    depths = np.full(len(cat), np.nan)
                    diagnostics = np.full(len(cat), np.nan)
                if len(aper_diams) == 1:
                    band_depths = list(depths)
                    band_diagnostics = list(diagnostics)
                    band_sigmas = list(funcs.n_sigma_detection(
                        depths, cat[f"MAG_APER_{band_data.filt_name}"], band_data.ZP
                        ))
                else:
                    if j == 0:
                        band_depths = [(depth,) for depth in depths]
                        band_diagnostics = [
                            (diagnostic,) for diagnostic in diagnostics
                        ]
                        band_sigmas = [
                            (
                                funcs.n_sigma_detection(
                                    depth, mag_aper[0], band_data.ZP
                                ),
                            )
                            for depth, mag_aper in zip(
                                depths, cat[f"MAG_APER_{band_data.filt_name}"]
                            )
                        ]
                    else:
                        band_depths = [
                            band_depth + (aper_diam_depth,)
                            for band_depth, aper_diam_depth in zip(
                                band_depths, depths
                            )
                        ]
                        band_diagnostics = [
                            band_diagnostic + (aper_diam_diagnostic,)
                            for band_diagnostic, aper_diam_diagnostic in zip(
                                band_diagnostics, diagnostics
                            )
                        ]
                        band_sigmas = [
                            band_sigma
                            + (
                                funcs.n_sigma_detection(
                                    depth, mag_aper[j], band_data.ZP
                                ),
                            )
                            for band_sigma, depth, mag_aper in zip(
                                band_sigmas, depths, cat[f"MAG_APER_{band_data.filt_name}"]
                            )
                        ]

            # update band with depths and diagnostics
            cat[f"loc_depth_{band_data.filt_name}"] = band_depths
            cat[f"{diagnostic_name}_{band_data.filt_name}"] = band_diagnostics
            cat[f"sigma_{band_data.filt_name}"] = band_sigmas
            # make local depth error columns in image units
            if len(aper_diams) == 1:
                cat[f"FLUXERR_APER_{band_data.filt_name}_loc_depth"] = \
                    list(funcs.mag_to_flux(np.array(band_depths), band_data.ZP) / 5.0)
            else:
                cat[f"FLUXERR_APER_{band_data.filt_name}_loc_depth"] = [
                    tuple(
                        [
                            funcs.mag_to_flux(val, band_data.ZP) / 5.0
                            for val in element
                        ]
                    )
                    for element in band_depths
                ]

            # impose n_pc min flux error and convert to Jy where appropriate
            if len(aper_diams) == 1:
                # TODO: Speed up this bit of code
                cat[f"FLUXERR_APER_{band_data.filt_name}_loc_depth_{str(int(min_flux_pc_err))}pc_Jy"] = \
                    [
                        np.nan
                        if flux == 0.0
                        else funcs.flux_image_to_Jy(
                            flux, band_data.ZP
                        ).value
                        * min_flux_pc_err
                        / 100.0
                        if err / flux
                        < min_flux_pc_err / 100.0
                        and flux > 0.0
                        else funcs.flux_image_to_Jy(
                            err, band_data.ZP
                        ).value
                        for flux, err in zip(
                            cat[f"FLUX_APER_{band_data.filt_name}_aper_corr"],
                            cat[f"FLUXERR_APER_{band_data.filt_name}_loc_depth"],
                        )
                    ]
            else:
                cat[f"FLUXERR_APER_{band_data.filt_name}_loc_depth_{str(int(min_flux_pc_err))}pc_Jy"] = \
                    [
                    tuple(
                        [
                            np.nan
                            if flux == 0.0
                            else funcs.flux_image_to_Jy(
                                flux, band_data.ZP
                            ).value
                            * min_flux_pc_err
                            / 100.0
                            if err / flux
                            < min_flux_pc_err / 100.0
                            and flux > 0.0
                            else funcs.flux_image_to_Jy(
                                err, band_data.ZP
                            ).value
                            for flux, err in zip(flux_tup, err_tup)
                        ]
                    )
                    for flux_tup, err_tup in zip(
                        cat[f"FLUX_APER_{band_data.filt_name}_aper_corr"],
                        cat[f"FLUXERR_APER_{band_data.filt_name}_loc_depth"],
                    )
                ]
        # update meta
        cat.meta = {**cat.meta, "MINPCERR": min_flux_pc_err}

        # overwrite original catalogue with local depth columns
        cat.write(self.phot_cat_path, overwrite=True)
        funcs.change_file_permissions(self.phot_cat_path)
        galfind_logger.info(
            f"Appended local depth columns to {self.phot_cat_path}"
        )
    else:
        galfind_logger.info(
            f"Local depth columns already exist in {self.phot_cat_path}"
        )

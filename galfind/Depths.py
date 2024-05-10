# Depths.py

#import automask as am
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.visualization as vis
from photutils.aperture import CircularAperture
from scipy.ndimage import uniform_filter
from tqdm import tqdm as tq
from matplotlib import cm
from astropy.table import Table
import cv2 as cv2
from galfind import Data
from kneed import KneeLocator
from astropy.io import fits
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from scipy import ndimage
from skimage import morphology
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from numba import jit
from kneed import KneeLocator
# install cv2, skimage, sklearn 


def do_photometry(image, xy_coords, radius_pixels):
    aper = CircularAperture(xy_coords, radius_pixels)
    aper_sums, _  = aper.do_photometry(image, error=None)
    return aper_sums


def make_grid(data, mask, radius, scatter_size, pixel_scale=0.03, plot=False, ax=None, distance_to_mask=50):
    '''
    data: 2D numpy array
    radius: float in arcseconds
    scatter_size: float in arcseconds
    pixel_scale: float in arcseconds/pixel
    '''
    radius_pixels = radius / pixel_scale
    scatter_size_pixels = scatter_size / pixel_scale

    #assert radius > scatter_size, "Radius must be greater than scatter size"
    # NOTE!!!!
    # np.shape on a 2D array returns (y, x) not (x, y)
    # So references to an x, y coordinate in the array should be [y, x]

    xy = np.mgrid[radius_pixels + scatter_size_pixels:data.shape[1]-(radius_pixels + scatter_size_pixels):2*(radius_pixels + scatter_size_pixels), 
                radius_pixels + scatter_size_pixels:data.shape[0]-(radius_pixels + scatter_size_pixels):2*(radius_pixels+scatter_size_pixels)]
    

    xy = xy.reshape(2, -1).T

    scatter = np.random.uniform(low=-scatter_size_pixels, high=scatter_size_pixels, size=(xy.shape[0], 2))
    
    xy = xy + scatter
    #print('Img shape, mask shape, max x grid, max y grid')
    #print(data.shape, mask.shape, np.max(xy[:, 0]), np.max(xy[:, 1]))
    
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance_to_mask, distance_to_mask)) #set up a circle of radius distance_to_mask pixels to mask around location of 0's
    mask = cv2.dilate(mask, kernel, iterations = 1) #dilate mask using the circle

    mask = mask.astype(bool)

    non_overlapping_xy = []
    for x, y in xy:
        # Check if the circle overlaps with the mask
        if np.any(mask[int(y-radius_pixels):int(y+radius_pixels), int(x-radius_pixels):int(x+radius_pixels)]):
            continue  # Skip this coordinate if it overlaps
        else:
            non_overlapping_xy.append((x, y))  # Add non-overlapping coordinates to the list
    # Plot the circles using matplotlib
    if plot:
        if ax == None:
            fig, ax = plt.subplots()

        stretch = vis.CompositeStretch(vis.LogStretch(), vis.ContrastBiasStretch(contrast=30, bias=0.08))    
        norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)

        ax.imshow(data, cmap='Greys', origin='lower', interpolation='None',
        norm=norm)

        for x, y in non_overlapping_xy:
            circle = plt.Circle((x, y), radius_pixels, color='r', fill=True)
            ax.add_artist(circle)
    
    return non_overlapping_xy
    

def calc_depths(coordinates, fluxes, img_data, mask = None, catalogue = None, 
                mode='rolling', sigma_level = 5, step_size=100, region_radius_used_pix=300, 
                zero_point = 28.08, min_number_of_values=100, n_nearest=100, split_depths = False, wht_data = None,
                n_split = 2, split_depth_min_size = 100000, split_depths_factor = 5,
                coord_type = 'sky', wcs = None, provide_labels=None,
                diagnostic_id = None, plot = False):
    '''
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
    '''
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
    if type(wht_data) != type(None) and split_depths:
        if type(provide_labels) == type(None):
            print('Obtaining labels...')
            assert np.shape(wht_data) == np.shape(img_data), f'The weight map must have the same shape as the image {np.shape(wht_data)} != {np.shape(img_data)}'
            labels_final, weight_map_smoothed = cluster_wht_map(wht_data, num_regions = n_split, bin_factor = split_depths_factor, min_size = split_depth_min_size)
            print('Labels obtained')
        else:
            labels_final = provide_labels
            print('Using provided labels')
    else:   
        print('Not labelling data')
        labels_final = np.zeros_like(img_data)

    #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
    # So references to an x, y coordinate in the array should be [y, x]
    
    if type(catalogue) == type(None):
        # If the catalogue is not provided, use the grid mode
        iterate_mode = 'grid'
    else:
        # If the catalogue is provided, use the catalogue mode 
        iterate_mode = 'catalogue'
        # Correct the coordinates if they are in sky coordinates

        if coord_type == 'sky' and wcs != None:
            # This doesn't work because footprint of the image is not the same as the footprint of the catalogue
            cat_x_col, cat_y_col = "ALPHA_J2000", "DELTA_J2000"
            ra_pix, dec_pix = wcs.all_world2pix(catalogue[cat_x_col], catalogue[cat_y_col], 0)
            cat_x, cat_y = ra_pix, dec_pix
            if type(wht_data) != type(None):
                assert np.shape(wht_data) == np.shape(img_data)

        elif coord_type == 'pixel':
            cat_x_col, cat_y_col = "X_IMAGE", "Y_IMAGE"
            cat_x, cat_y = catalogue[cat_x_col], catalogue[cat_y_col]
        else:
            raise ValueError('coord_type must be either "sky" or "pixel"')
            
    #print('Image dimensions:', np.shape(img_data))
   
    x_max, y_max = np.max(x), np.max(y)
    x_label, y_label = x.astype(int), y.astype(int)
    # Don't look for label of pixels outside the image
    #print('X label max:', np.max(x_label), 'Y label max:', np.max(y_label))
    #print('before clip')
    x_label = np.clip(x_label, 0, x_max - 1).astype(int)
    y_label = np.clip(y_label, 0, y_max - 1).astype(int)

    #print('X max:', x_max, 'Y max:', y_max)
    #print('X label max:', np.max(x_label), 'Y label max:', np.max(y_label))
    
    # Create an empty grid for Y values
    #flux_grid = np.zeros(grid_size)
    #flux_grid[:] = np.nan
    # Assign Y values to the grid based on coordinates
    #flux_grid[x.astype(int), y.astype(int)] = fluxes
    # Apply 2D rolling average filter
    # smoothed_nmad_flux = np.zeros_like(flux_grid)

    # i is the x coordinate, j is the y coordinate
    if iterate_mode == 'grid':

        grid_size = (int(np.ceil(x_max)) + step_size, int(np.ceil(y_max)) + step_size)

        #print(f'i max: {grid_size[0]}, j max: {grid_size[1]}')

        nmad_sized_grid = np.zeros((grid_size[1]//step_size + 1, grid_size[0]//step_size  + 1) )
        num_sized_grid =  np.zeros((grid_size[1]//step_size + 1, grid_size[0]//step_size  + 1) )
        num_sized_grid[:] = np.nan
        label_size_grid = np.zeros((grid_size[1]//step_size + 1, grid_size[0]//step_size  + 1) )
        label_size_grid[:] = np.nan
        #print('Grid size:', grid_size)
        for i in tq(range(0, grid_size[0], step_size)):
            for j in range(0, grid_size[1], step_size):
                setnan = False
                j_label = np.clip(j, 0, y_max - 1)
                i_label = np.clip(i, 0, x_max - 1)
                # print(j_label, grid_size[1], i_label, grid_size[0], labels_final.shape)
                
                #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
                # So references to an x, y coordinate in the array should be [y, x]
    
                label = labels_final[int(j_label), int(i_label)]
                label_name = label
                if type(mask) != type(None):
                    # Don't calculate the depth if the coordinate is masked
                    try:
                        #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
                        # So references to an x, y coordinate in the array should be [y, x]
                        if mask[j, i] == 1.0:
                            depth = np.nan
                            setnan = True
                            num_of_apers = np.nan    
                    except IndexError:
                        setnan = True
                        depth = np.nan
                        num_of_apers = np.nan
                        
                if not setnan:
                    # Calculate the distance from the center
                    distances = np.sqrt((x - i)**2 + (y - j)**2)
                    
                    if mode == 'rolling':
                        # Extract the neighboring Y values within the circular window
                        # Ensure label values of regions are the same as label
                        neighbor_values = fluxes[(distances <= region_radius_used_pix) & (labels_final[y_label, x_label] == label)]
                        #neighbor_values = fluxes[distances <= region_radius_used_pix]
                        # Calculate the NMAD of the neighboring Y values
                    elif mode == 'n_nearest':
                        # Extract the n nearest Y values
                        # Ensure label values of regions are the same as label
                        distances = distances[labels_final[y_label, x_label] == label]
                        fluxes_i = fluxes[labels_final[y_label, x_label] == label]
                        neighbor_values = fluxes[np.argsort(distances)[:n_nearest]]
                      
                        min_number_of_values = n_nearest
                    num_of_apers = len(neighbor_values)

                    depth = calculate_depth(neighbor_values, sigma_level, zero_point, min_number_of_values=min_number_of_values)
                
                # NOTE np.shape on a 2D array returns (y, x) not (x, y)
                # So references to an x, y coordinate in the array should be [y, x]
                num_sized_grid[j//step_size, i//step_size] = num_of_apers
                nmad_sized_grid[j//step_size, i//step_size] = depth
                label_size_grid[j//step_size, i//step_size] = label_name

        if plot:
            plt.imshow(nmad_sized_grid, origin='lower', interpolation='None', cmap='plasma')
            plt.show()
        
        return nmad_sized_grid, num_sized_grid, label_size_grid, labels_final

    elif iterate_mode == 'catalogue':
        depths, diagnostic, cat_labels = [], [], []
        count = 0
        for i, j in tq(zip(cat_x, cat_y), total = len(cat_x)):
            # Check if the coordinate is outside the image or in the mask
            if (i > x_max or i < 0 or j > y_max or j < 0):
                depth = np.nan
                num_of_apers = np.nan
                label = np.nan
                depth_diagnostic = np.nan
            else:
                label = labels_final[int(j), int(i)]
                distances = np.sqrt((x - i)**2 + (y - j)**2)
                # NOTE np.shape on a 2D array returns (y, x) not (x, y)
                # So references to an x, y coordinate in the array should be [y, x]

                distances_i = distances[labels_final[y_label, x_label] == label]
                fluxes_i = fluxes[labels_final[y_label, x_label] == label]

                if mode == 'rolling':
                    # Extract the neighboring Y values within the circular window

                    neighbor_values = fluxes_i[distances_i <= region_radius_used_pix]
                    # Calculate the NMAD of the neighboring Y values
                    # Diagnostic is number of apertures used
                    depth_diagnostic = len(neighbor_values)
                elif mode == 'n_nearest':
                    # Extract the n nearest Y values
                    fluxes_i = fluxes[labels_final[y_label, x_label] == label]
                    indexes = np.argsort(distances_i)
                    #print(i, j, label, distances[indexes][:n_nearest], neighbor_values)
                    neighbor_values = fluxes_i[indexes][:n_nearest]
                    min_number_of_values = n_nearest
                    # Diagnostic is the distance to the n_nearest neighbor
                    depth_diagnostic = distances_i[indexes][:n_nearest][-1]

                    if plot:
                        if count == diagnostic_id:
            
                            # Plot regions used and image
                            fig, ax = plt.subplots()
                            ax.imshow(img_data, cmap='Greys', origin='lower', interpolation='None')

                            # Do this with matplotlib instead
                            circle = plt.Circle((i, j), radius_pixels, color='r', fill=False)
                            ax.add_artist(circle)
                            xtest, ytest = x[labels_final[y_label, x_label] == label], y[labels_final[y_label, x_label] == label]
                            xtest = xtest[indexes][:n_nearest]
                            ytest = ytest[indexes][:n_nearest]
                            for (xi, yi) in zip(xtest, ytest):
                                circle = plt.Circle((xi, yi), radius_pixels, color='b', fill=False)
                                ax.add_artist(circle)
                            plt.show()

                depth = calculate_depth(neighbor_values, sigma_level, zero_point, min_number_of_values=min_number_of_values)

            cat_labels.append(label)
            depths.append(depth)
            diagnostic.append(depth_diagnostic)
            count += 1
        
        return np.array(depths), np.array(diagnostic), np.array(cat_labels), labels_final

def show_depths(nmad_grid, num_grid, step_size, region_radius_used_pix, labels = None,
                cat_labels = None, cat_depths = None, cat_diagnostics = None, 
                x_pix = None, y_pix = None, img_mask = None, labels_final = None, \
                suptitle = None, save_path = None, show = False):
    fig, axs = plt.subplots(3 if type(labels) == type(None) else 4, 1 if type(cat_depths) == type(None) else 2, facecolor = 'white', figsize = (8, 14))
    axs = axs.flatten()
    if type(suptitle) != type(None):
        fig.suptitle(suptitle, fontsize='large', fontweight='bold')
    cmap = cm.get_cmap("plasma")
    cmap.set_bad(color='black')
    # Make vmin and vmax the 5th and 95th percentile of the nmad_grid
    vmin, vmax = np.nanpercentile(nmad_grid, 1), np.nanpercentile(nmad_grid, 99)
    mappable = axs[0].imshow(nmad_grid, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    radius = region_radius_used_pix/step_size
    circle_x = radius + 0.07 * np.shape(nmad_grid)[1]
    circle_y = radius + 0.07 * np.shape(nmad_grid)[0]
   
    print('Circle x, y:', circle_x, circle_y)
    patch = plt.Circle((circle_x, circle_y), radius = radius, fill=True, facecolor='white', lw=2, zorder=10)
    axs[0].add_patch(patch)
    axs[0].text(circle_x, circle_y - 1.4*radius, s='Filter Size', va='center', ha='center', color='white', fontsize='medium')# path_effects = [pe.Stroke(linewidth=0.5, foreground='black')])
    fig.colorbar(mappable, label=r'5$\sigma$ Depth',  ax=axs[0])
    axs[0].set_title('Rolling Average 5$\sigma$ Depth')

    mappable2 = axs[1].imshow(num_grid, origin='lower', cmap=cmap)
    fig.colorbar(mappable2, label=r'Number of Apertures Used',  ax=axs[1])
    axs[1].set_title('Rolling Average Diagnostic')      
    
    if type(labels) != type(None):
        cmap = cm.get_cmap('Set2')
        if len(np.unique(labels)) > 1:
            num_labels = len(np.unique(labels))
            custom_cmap = LinearSegmentedColormap.from_list('custom', [cmap(i/num_labels) for i in range(num_labels)], num_labels)
            pos = 5 if type(cat_depths) != type(None) else 2
            mappable = axs[pos+1].imshow(labels, cmap=custom_cmap, origin='lower', interpolation='None')

            possible_labels = np.unique(labels)
            av_depths = [np.nanmedian(nmad_grid[labels == label]) for label in possible_labels]
            print('Average depths:', av_depths)
            print(possible_labels)
            order = np.argsort(av_depths)
            possible_labels = possible_labels[order]
            
            label = ['Shallow', 'Deep']
            colors = [custom_cmap(possible_labels[0]),custom_cmap(possible_labels[1])]
            axs[pos+1].set_title('Labels')

            axs[pos].set_title('Catalogue Labels')

            axs[pos].imshow(img_mask, cmap='Greens', origin='lower', interpolation='None', alpha=0.3, zorder=4)
            axs[pos].imshow(labels_final, cmap='Reds', origin='lower', interpolation='None', alpha=0.3, zorder=4)
            m = axs[pos].scatter(x_pix, y_pix, s=1, zorder=5, c = np.array(cat_labels), cmap='plasma')
            #fig.colorbar(ax=axs[5], mappable=m, label='Label')
        else:
            label = ['Single Region']
            possible_labels = [0]
            colors = [cmap(0)]
            axs[-2].remove()
            axs[-1].remove()
        
    # Histogram of depths
    if type(cat_depths) != type(None):
        #plt.scatter(x_pix, y_pix, s=1, zorder=5, c = depth, cmap='plasma')

        axs[2].set_title('Catalogue Depths')
        m = axs[2].scatter(x_pix, y_pix, s=1, zorder=5, c = cat_depths, cmap='plasma')
        fig.colorbar(ax=axs[2], mappable=m, label='5$\sigma$ Depth')

        axs[2].imshow(img_mask, cmap='Greens', origin='lower', interpolation='None', alpha=0.3, zorder=4)
        axs[2].imshow(labels_final, cmap='Reds', origin='lower', interpolation='None', alpha=0.3, zorder=4)

        axs[4].set_title('Catalogue Diagnostic')
        m = axs[4].scatter(x_pix, y_pix, s=1, zorder=5, c = np.array(cat_diagnostics), cmap = 'plasma')
        fig.colorbar(ax=axs[4], mappable=m, label='Distance to 200th Nearest Empty Aperture')
        
        set_labels = [cat_depths[cat_labels == label] for label in possible_labels]
        axs[3].hist(set_labels, bins=40, range=(np.nanmin(cat_depths), np.nanmax(cat_depths)), label=label, color=colors, histtype='stepfilled', alpha=0.8)    
        # Plot line at median depth
        # Fix y range
        axs[3].set_ylim(axs[3].get_ylim())
        max = axs[3].get_ylim()[1]
        
        for pos, depth in enumerate(set_labels):
            axs[3].axvline(np.nanmedian(depth), 0, max, color='black', lw=3, linestyle='--', label='Median' if pos == 0 else '', zorder=10)
            axs[3].axvline(np.nanmean(depth), 0, max, color='black', lw=3, linestyle='-', label='Mean' if pos == 0 else '', zorder=10)
            # Label with text
            axs[3].text(np.nanmean(depth), 0.7*max, f'{np.nanmean(depth):.2f}', va='top', ha='center', fontsize='medium', color=colors[pos], rotation=90, path_effects = [pe.withStroke(linewidth=3, foreground='white')], zorder=10, fontweight='bold')
            axs[3].text(np.nanmedian(depth), 0.9*max, f'{np.nanmedian(depth):.2f}', va='top', ha='center', fontsize='medium', color=colors[pos], rotation=90, path_effects = [pe.withStroke(linewidth=3, foreground='white')], zorder=10, fontweight='bold')
        
        axs[3].set_xlabel('5$\sigma$ Depth')
        axs[3].set_title('Depth Histogram')
        axs[3].legend(frameon=False)

    else:
        print('No catalogue depths')
    #axs[7].remove()
    #plt.tight_layout()
    if type(save_path) != type(None):
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved depths plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.clf()

    print('Median 5 sigma depth:', np.nanmedian(nmad_grid))
    print('Median number of apertures used:', np.nanmedian(num_grid))
    print('Mean 5 sigma depth:', np.nanmean(nmad_grid))
    return fig, axs

def calc_depths_numba(coordinates, fluxes, img_data, mask = None, catalogue = None, 
                mode='rolling', sigma_level = 5, step_size=100, region_radius_used_pix=300, 
                zero_point = 28.08, min_number_of_values=100, n_nearest=100, split_depths = False, wht_data = None,
                n_split = 1, split_depth_min_size = 100000, split_depths_factor = 5,
                coord_type = 'sky', wcs = None, provide_labels=None,
                diagnostic_id = None, plot = False):
    '''
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
    '''
    print('This is the experimental numba version')
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
    if type(wht_data) != type(None) and split_depths:
        if type(provide_labels) == type(None):
            print('Obtaining labels...')
            assert np.shape(wht_data) == np.shape(img_data), f'The weight map must have the same shape as the image {np.shape(wht_data)} != {np.shape(img_data)}'
            labels_final, weight_map_smoothed = cluster_wht_map(wht_data, num_regions = n_split, bin_factor = split_depths_factor, min_size = split_depth_min_size)
            print('Labels obtained')
        else:
            labels_final = provide_labels
            print('Using provided labels')
    else:   
        print('Not labelling data')
        labels_final = np.zeros_like(img_data, dtype=np.float64)

    #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
    # So references to an x, y coordinate in the array should be [y, x]
    
    if type(catalogue) == type(None):
        # If the catalogue is not provided, use the grid mode
        iterate_mode = 'grid'
    else:
        # If the catalogue is provided, use the catalogue mode 
        iterate_mode = 'catalogue'
        # Correct the coordinates if they are in sky coordinates

        if coord_type == 'sky' and wcs != None:
            # This doesn't work because footprint of the image is not the same as the footprint of the catalogue
            cat_x_col, cat_y_col = "ALPHA_J2000", "DELTA_J2000"
            ra_pix, dec_pix = wcs.all_world2pix(catalogue[cat_x_col], catalogue[cat_y_col], 0)
            cat_x, cat_y = ra_pix, dec_pix
            if type(wht_data) != type(None):
                assert np.shape(wht_data) == np.shape(img_data)

        elif coord_type == 'pixel':
            cat_x_col, cat_y_col = "X_IMAGE", "Y_IMAGE"
            cat_x, cat_y = catalogue[cat_x_col], catalogue[cat_y_col]
        else:
            raise ValueError('coord_type must be either "sky" or "pixel"')
            
    x_max, y_max = np.max(x), np.max(y)
    x_label, y_label = x.astype(int), y.astype(int)
    # Don't look for label of pixels outside the image

    x_label = np.clip(x_label, 0, x_max - 1).astype(int)
    y_label = np.clip(y_label, 0, y_max - 1).astype(int)

    filter_labels = labels_final[y_label, x_label] #.astype(np.float64)

    # i is the x coordinate, j is the y coordinate
    print('Iterate mode:', iterate_mode)
    if iterate_mode == 'grid':

        grid_size = (int(np.ceil(x_max)) + step_size, int(np.ceil(y_max)) + step_size)

        nmad_sized_grid = np.zeros((grid_size[1]//step_size + 1, grid_size[0]//step_size  + 1) )
        num_sized_grid =  np.zeros((grid_size[1]//step_size + 1, grid_size[0]//step_size  + 1) )
        num_sized_grid[:] = np.nan
        label_size_grid = np.zeros((grid_size[1]//step_size + 1, grid_size[0]//step_size  + 1) )
        label_size_grid[:] = np.nan
        #print('Grid size:', grid_size)
        for i in tq(range(0, grid_size[0], step_size)):
            for j in range(0, grid_size[1], step_size):
                setnan = False
                if type(mask) != type(None):
                    # Don't calculate the depth if the coordinate is masked
                    try:
                        #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
                        # So references to an x, y coordinate in the array should be [y, x]
                        if mask[j, i] == 1.0:
                            depth = np.nan
                            setnan = True
                            num_of_apers = np.nan    
                    except IndexError:
                        setnan = True
                        depth = np.nan
                        num_of_apers = np.nan
                
                
                if not setnan:
                    j_label = np.clip(j, 0, y_max - 1)
                    i_label = np.clip(i, 0, x_max - 1)
                    #  NOTE np.shape on a 2D array returns (y, x) not (x, y)
                    # So references to an x, y coordinate in the array should be [y, x]            
                    label = labels_final[j_label.astype(int), i_label.astype(int)]
                    distances = numba_distances(x, y, i, j)
                    label_name = label
                    if mode == 'rolling':
                        # Extract the neighboring Y values within the circular window
                        # Ensure label values of regions are the same as label
                        neighbor_values = fluxes[(distances <= region_radius_used_pix) & (filter_labels == label)]
                        #neighbor_values = fluxes[distances <= region_radius_used_pix]
                        
                        # Calculate the NMAD of the neighboring Y values
                    elif mode == 'n_nearest':
                        # Extract the n nearest Y values
                        # Ensure label values of regions are the same as label
                        
                        # Create a boolean mask
                        mask = filter_labels == label
                        distances_i = distances[mask]
                        fluxes_i = fluxes[mask]

                        nearest_indices = np.argpartition(distances, n_nearest)[:n_nearest]
                        neighbor_values = fluxes[nearest_indices]
                        min_number_of_values = n_nearest
                        
                    num_of_apers = len(neighbor_values)

                    depth = calculate_depth(neighbor_values, sigma_level, zero_point, min_number_of_values=min_number_of_values)
                
                # NOTE np.shape on a 2D array returns (y, x) not (x, y)
                # So references to an x, y coordinate in the array should be [y, x]
                num_sized_grid[j//step_size, i//step_size] = num_of_apers
                nmad_sized_grid[j//step_size, i//step_size] = depth
                label_size_grid[j//step_size, i//step_size] = label_name

        if plot:
            plt.imshow(nmad_sized_grid, origin='lower', interpolation='None', cmap='plasma')
            plt.show()
        
        return nmad_sized_grid, num_sized_grid, label_size_grid, labels_final

    elif iterate_mode == 'catalogue':
        depths, diagnostic, cat_labels = [], [], []
        count = 0
        print('Total number', len(cat_x))
        for i, j in tq(zip(cat_x, cat_y), total = len(cat_x)):
            # Check if the coordinate is outside the image or in the mask
            if (i > x_max or i < 0 or j > y_max or j < 0):
                depth = np.nan
                num_of_apers = np.nan
                label = np.nan
                depth_diagnostic = np.nan
            else:
                label = labels_final[j.astype(int), i.astype(int)]
                
                #distances = np.sqrt((x - i)**2 + (y - j)**2)
                # NOTE np.shape on a 2D array returns (y, x) not (x, y)
                # So references to an x, y coordinate in the array should be [y, x]
               
                distances = numba_distances(x, y, i, j)
                # Get the label of interest
                label = labels_final[j.astype(int), i.astype(int)]
            
                # Create a boolean mask
                mask = filter_labels == label

                distances_i = distances[mask]
                fluxes_i = fluxes[mask]

                if mode == 'rolling':
                    neighbor_values = fluxes[(distances <= region_radius_used_pix) & (labels_final[y_label, x_label] == label)]
                    depth_diagnostic = len(neighbor_values)
                    
                elif mode == 'n_nearest':

                    #print(labels_final.dtype, y_label.dtype, x_label.dtype, fluxes.dtype, n_nearest.dtype, label.dtype)
                    #neighbor_values, depth_diagnostic = numba_n_nearest_filter(fluxes_i, distances_i, n_nearest)
                    nearest_indices = np.argpartition(distances, n_nearest)[:n_nearest]
                    neighbor_values = fluxes[nearest_indices]

                    min_number_of_values = n_nearest
                    # Depth diagnostic is distance to n_nearest
                    depth_diagnostic = nearest_indices

                    if plot:
                        if count == diagnostic_id:
            
                            # Plot regions used and image
                            fig, ax = plt.subplots()
                            ax.imshow(img_data, cmap='Greys', origin='lower', interpolation='None')

                            # Do this with matplotlib instead
                            circle = plt.Circle((i, j), radius_pixels, color='r', fill=False)
                            ax.add_artist(circle)
                            xtest, ytest = x[labels_final[y_label, x_label] == label], y[labels_final[y_label, x_label] == label]
                            xtest = xtest[indexes][:n_nearest]
                            ytest = ytest[indexes][:n_nearest]
                            for (xi, yi) in zip(xtest, ytest):
                                circle = plt.Circle((xi, yi), radius_pixels, color='b', fill=False)
                                ax.add_artist(circle)
                            plt.show()
                depth = calculate_depth(neighbor_values, sigma_level, zero_point, min_number_of_values=min_number_of_values)
                #print(f'Fractional time for labelling: {(end_labelling - start_labelling)/(end_overall - start_overall)}')
                #print(f'Fractional time for distances: {(end_distances - start_distances)/(end_overall - start_overall)}')
                #print(f'Fractional time for filtering: {(end_filtering - start_filtering)/(end_overall - start_overall)}')
                #print(f'Fractional time for indices: {(end_indices - start_indices)/(end_overall - start_overall)}')
                #print(f'Fractional time for label2:, {(end_label2 - start_label2)/(end_overall - start_overall)}')
                #print('\n')    
            
            cat_labels.append(label)
            depths.append(depth)
            diagnostic.append(depth_diagnostic)
            count += 1
            
        return np.array(depths), np.array(diagnostic), np.array(cat_labels), labels_final

@jit(nopython=True)
def numba_distances(x=np.array([]), y=np.array([]), x_coords=1, y_coords=1):
    #distances = np.sqrt((x_coords[:, None] - x)**2 + (y_coords[:, None]- y)**2)
    #return distances
    distances = np.zeros_like(x)
    for i in range(len(x)):
        distances[i] = np.sqrt((x[i] - y_coords)**2 + (y[i] - y_coords)**2)
    return distances

@jit(nopython=True)
def calculate_depth(values, sigma_level = 5, zero_point = 28.08, min_number_of_values = 100):
    if len(values) < min_number_of_values:
        return np.nan
    median = np.nanmedian(values)
    abs_deviation = np.abs(values - median)
    nmad = 1.4826 * np.nanmedian(abs_deviation) * sigma_level
    if nmad > 0.:
        depth_sigma = -2.5 * np.log10(nmad) + zero_point
    else:
        depth_sigma = np.nan
    return depth_sigma

def make_ds9_region_file(coordinates, radius, filename, coordinate_type = 'sky', convert=True, wcs=None, pixel_scale=0.03):
    '''
    coordinates: list of tuples - (x, y) coordinates
    radius: float - the radius of the circles in units of sky or pixels
    filename: str - the name of the file to write the regions to
    coordinate_type: str - 'sky' or 'pixel'
    convert: bool - whether to convert the coordinates to the other coordinate type
    wcs = WCS object - the wcs object to use for the conversion if convert is True
    '''
    # If coordinate shape is (2, n) then we have to transpose it
    if np.shape(coordinates)[-1] == 2:
        coordinates = np.array(coordinates).T
    print(f"empty aperture coordinates = {coordinates}")
    x, y = np.array(coordinates)
    
    if coordinate_type == 'sky':
        coord_type = 'fk5'
        radius_unit = '"'
        if convert:
            x, y = wcs.all_pix2world(x, y, 0)
            radius = radius * pixel_scale
   
    elif coordinate_type == 'pixel':
        coord_type = 'image'
        radius_unit = ''
        if convert:
            x, y = wcs.all_world2pix(x, y, 0)
            radius = radius / pixel_scale
        
    with open(filename, 'w') as f:
        f.write(f'# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n{coord_type}\n')
        for xi, yi in zip(x, y):
            f.write(f'circle({xi},{yi},{radius:.5f}{radius_unit})\n')


def cluster_wht_map(wht_map, num_regions='auto', bin_factor=1, min_size=10000, plot=False):
    'Works best for 2 regions, but can be used for more than 2 regions - may need additional smoothing and cleaning'
    # Read the image and associated weight map

    # adjust min_size to be in terms of the bin_factor
    min_size = min_size // bin_factor**2
    
    if type(wht_map) == str:
        # 
        weight_map = fits.open(wht_map)
        # Check if we have multiple extensions
        if len(weight_map) > 1:
            weight_map = weight_map['WHT'].data
        else:
            weight_map = weight_map[0].data
            
    elif type(wht_map) == np.ndarray:
        weight_map = wht_map


    # Remove NANs
    weight_map[np.isnan(weight_map)] = 0
    percentiles = np.nanpercentile(weight_map, [5, 95])

    weight_map_clipped = np.clip(weight_map, percentiles[0], percentiles[1])
    weight_map_transformed = (weight_map_clipped  - percentiles[0]) / (percentiles[1] - percentiles[0]) * 255
    
    weight_map_smoothed = cv2.resize(weight_map_transformed, (weight_map.shape[1]//bin_factor, weight_map.shape[0]//bin_factor), interpolation=cv2.INTER_LINEAR)
    # Renormalize
    
    #weight_map_smoothed = cv2.normalize(weight_map_smoothed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    percentiles = np.nanpercentile(weight_map_smoothed, [5, 95])
    weight_map_clipped = np.clip(weight_map_smoothed, percentiles[0], percentiles[1])
    weight_map_transformed = (weight_map_clipped  - percentiles[0]) / (percentiles[1] - percentiles[0]) * 255
    
    labels_filled = []
    iterations = 0
    if num_regions == 'auto':
        num_regions_list = [1, 2, 3, 4]
    else:
        num_regions_list = [num_regions]
    sse = []
    if len(num_regions_list) > 1:
        for num_regions in num_regions_list:

            #while len(np.unique(labels_filled)) != num_regions:
                #suming you want to segment into 2 regions (deep and non-deep)
            print(num_regions)
            kmeans = KMeans(n_clusters=num_regions, n_init=5)
        
            kmeans.fit(weight_map_transformed.flatten().reshape(-1, 1))

            sse.append(kmeans.inertia_)

        if len(np.unique(labels_filled)) == 1:
            print('KMeans failed to find regions. No regions used.')
            return np.zeros_like(weight_map), weight_map_smoothed
        from kneed import KneeLocator
                
        kneedle = KneeLocator(num_regions_list, sse, curve='convex', direction='decreasing')
        num_regions = kneedle.elbow
        print(f'Detected {num_regions} regions as best.')

        if plot:
            plt.plot(num_regions_list, sse)
            plt.xlabel('Number of Regions')
            plt.ylabel('SSE')
            plt.axvline(num_regions, color='red', linestyle='--')
            plt.show()
            plt.close()

        # Find best of doing it 15x
    kmeans = KMeans(n_clusters=num_regions, n_init=15)
    kmeans.fit(weight_map_transformed.flatten().reshape(-1, 1))
    
    labels = kmeans.labels_.reshape(weight_map_transformed.shape[:2])
    
    if num_regions == 2:
        # Closing and opening to remove light and dark spots
        labels_filled = morphology.binary_closing(labels, morphology.disk(5))
        labels_filled = morphology.binary_opening(labels_filled, morphology.disk(5))
        
    else:
        # Do this when you have more than 2 regions - doesn't work quite as well at the edges
        labels_filled = morphology.area_closing(labels, area_threshold=min_size)
        labels_filled = morphology.area_opening(labels_filled, area_threshold=min_size)

    # Remove remaining holes
    possible_labels = np.unique(labels_filled)
    for label in possible_labels:
        region = labels_filled == label
        region_cleaned = morphology.remove_small_holes(region, area_threshold=min_size)
        labels_filled = np.where(region_cleaned, label, labels_filled)
    
    
    # Check if both labels are present
    possible_labels = np.unique(labels_filled)
    if len(possible_labels) != num_regions:
        print('One of the Kmeans labelled regions didn\'t survive cleaning.')
        num_regions = len(possible_labels)
    

    # Check if one of regions is background (i.e very close to zero)
    zero_levels = [np.count_nonzero(weight_map_smoothed[labels_filled == label] < 10) / np.count_nonzero(labels_filled == label) for label in possible_labels]
    print('Zero levels:', zero_levels)
    possible_background_label = np.argmax(zero_levels)
    background_label = possible_labels[possible_background_label]
    background_frac = zero_levels[possible_background_label]
    if background_frac > 0.80:
        print('Label', int(background_label), 'is background')
        if num_regions == 2:
            print('No other regions detected, so no need to break depths into regions.')
            labels_filled = np.zeros_like(labels_filled)

    #plt.imshow(weight_map_smoothed, cmap='Greys', origin='lower', interpolation='None')
    #plt.imshow(labels_filled, cmap='viridis', origin='lower', interpolation='None', alpha=0.7)
    # If bin_factor is greater than 1, enlarge the labels_filled to the original size
    if bin_factor > 1:
        labels_filled = cv2.resize(labels_filled.astype(np.uint8), (weight_map.shape[1], weight_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    show_labels = False
    if show_labels:
        plt.imshow(labels_filled, cmap='viridis', origin='lower', interpolation='None')
        plt.show()

    return labels_filled, weight_map_transformed

if __name__ == '__main__':
    
    plt.rcParams['figure.dpi'] = 300

    # General info
    survey = 'COSMOS-Web-0A'
    instruments = ['NIRCam']
    version = 'v11'
    filter = 'F277W'
    save_path = '/nvme/scratch/work/tharvey/scripts/automask/'

    depth_region_radius = 0.16 # arcseconds
    sigma_level = 5 # sigma level for depth calculation

    # Catalogue Info
    cat_x_col = 'ALPHA_J2000'
    cat_y_col = 'DELTA_J2000'
    coord_type = 'sky'

    # Mask Options 
    star_mask_mode = 'simple' # 'simple' or 'sophisticated' - whether diffraction spikes are rotated
    custom_mask = None # Path to custom mask (optional)
    write_mask = False # Whether to write out the mask to a fits file
    edge_mask_distance = 50 # distance to mask around the edge of the image (pixels) 

    # Aperture Grid Options
    distance_to_mask = 30 # minimum distance to mask for aperture placement (pixels)
    scatter_size = 0.1 # distance to scatter positions of apertures in grid (arcsec)
    plot = True # plot the grid of apertures

    # Split Depths Options
    split_depths = False # whether to split the depths into regions using KMeans clustering
    split_depth_regions = 2 # the number of regions to split the depths into
    split_depth_min_size = 100_000 # the minimum size of the regions
    split_depths_factor = 5 # the factor to use for the binning of the weight map - lower is more accurate but slower

    diagnostic_id = None # the position of a glaaxy in the catalogue to show the diagnostic plot

    # Depth Options
    step_size = 100 # 2D step size for depth plot
    # For rolling mode
    region_radius_used_pix = 300 # pixel radius for depth calculation when using the rolling mode
    min_number_of_values = 100 # minimum number of values within region_radius_used_pix required to calculate the depth when using the rolling mode
    # For n_nearest mode
    n_nearest = 200 # number of nearest neighbors to use for depth calculation when using the n_nearest mode

    data = Data.from_pipeline(survey, version = version, instruments = instruments, excl_bands=['f435W', 'f775W', 'f850LP'])

    image_path = data.im_paths[filter]
    wht_path = data.wht_paths[filter]
    seg_path = data.seg_paths[filter]
    zero_point = data.im_zps[filter]
    pixel_size = data.im_pixel_scales[filter]
    image_ext = data.im_exts[filter]

    print('Image path:', image_path)
    print('WHT path:', wht_path)
    print('SEG path:', seg_path)

    if version == 'v9':
        if survey in ['El-Gordo', 'CLIO', 'SMACS-0723', 'GLASS', 'MACS-0416']:
            cat_path  = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/{version}/NIRCam/{survey}/{survey}_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection.fits"
        elif survey in ['CEERS']:
            cat_path  = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/{version}/ACS_WFC+NIRCam/Combined/CEERS_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV_updated.fits"
        elif survey in ['NEP-1', 'NEP-2', 'NEP-3', 'NEP-4', 'NGDEEP', 'JADES-Deep-GS']:
            cat_path  = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/{version}/ACS_WFC+NIRCam/{survey}/{survey}_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection.fits"
        elif survey in ['JADES-3215']:
            cat_path  = '/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v11/NIRCam/JADES-3215/JADES-3215_MASTER_Sel-f277W+f356W+f444W_v11_loc_depth_masked_10pc_eazy_fsps_larson_matched_selection.fits'
        elif 'COSMOS' in survey:
            cat_path = f'/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v11/NIRCam/{survey}/{survey}_MASTER_Sel-F277W+F444W_v11.fits'
        else:
            cat_path  = None
    else:
        cat_path = None
        
    if cat_path != None:
        cat = Table.read(cat_path)
    else:
        cat = None

    radius_pixels = depth_region_radius / pixel_size

    # Generate mask
    img_data, img_mask, img_wcs = mask_image(image_path, seg_path=seg_path,
                                            write=write_mask, image_ext=image_ext,
                                            edge_mask_distance=edge_mask_distance)

    # Generate regions
    xy = make_grid(img_data, img_mask, radius=depth_region_radius, 
                    scatter_size=scatter_size, distance_to_mask=distance_to_mask,
                    plot=plot)
    print(f'{len(xy)} apertures placed')

    # Make ds9 region file of apertures for compatability and debugging

    make_ds9_region_file(xy, radius_pixels, f'{save_path}/{survey}_{version}_{filter}.reg', 
    coordinate_type='pixel', convert=False, wcs=img_wcs, pixel_scale=pixel_size)

    #Get fluxes in regions
    fluxes = do_photometry(img_data, xy, radius_pixels)

    # Calculate depths for catalogue
    if cat_path != None:
        depths, diagnostic, depth_labels, final_labels = calc_depths(xy, fluxes, img_data, img_mask, 
                                        region_radius_used_pix=region_radius_used_pix, step_size=step_size, 
                                        catalogue=cat, wcs=img_wcs, cat_x_col = cat_x_col, 
                                        cat_y_col = cat_y_col, coord_type=coord_type,
                                        mode='n_nearest', n_nearest=n_nearest, zero_point=zero_point, split_depths=split_depths,
                                        split_depth_regions=split_depth_regions, split_depth_min_size=split_depth_min_size,
                                        split_depths_factor=split_depths_factor, wht_data = wht_path, diagnostic_id=diagnostic_id)
        # Diagnostic plot comparing depths to previous depths
        fig, ax = plt.subplots()
        x = cat[f'loc_depth_{filter}'][:, 0]
        y = depths
        masknans = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x[masknans]
        y = y[masknans]

        print('Median depth difference:', np.nanmedian(y - x))
        z = np.vstack([x,y])

        z = gaussian_kde(z)(z)
        ax.scatter(x, y, s= 1, c=z, cmap='plasma')
        ax.set_xlabel(f'loc_depth_{filter} (current)')
        ax.set_ylabel(f'loc_depth_{filter} (new)')
        # plot 1:1 line
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.plot([0, 40], [0, 40], 'k--')
        plt.show()
    else:
        final_labels = None

    print('Final labels:', final_labels)
    # Calculate depth plot
    nmad_grid, num_grid, labels_grid, final_labels = calc_depths(xy, fluxes, img_data, img_mask,
                                    region_radius_used_pix=region_radius_used_pix, 
                                    step_size = step_size, mode = 'rolling', min_number_of_values = min_number_of_values,
                                    zero_point = zero_point, split_depths = split_depths,
                                    split_depth_regions = split_depth_regions, split_depth_min_size = split_depth_min_size,
                                    split_depths_factor = split_depths_factor, wht_data = wht_path, provide_labels = final_labels)

    # Show depth plot
    if cat_path is not None:
        cat_x, cat_y = cat[cat_x_col], cat[cat_y_col]
        x_pix, y_pix = img_wcs.all_world2pix(cat_x, cat_y, 0)
    else:
        x_pix, y_pix = None, None
        depth_labels = None
        depths = None
        diagnostic = None

    depths_fig, depths_ax = show_depths(nmad_grid, num_grid, step_size, region_radius_used_pix, 
        labels_grid, depth_labels, depths, diagnostic, x_pix, y_pix, 
        img_mask, final_labels, suptitle=f'{survey} {version} {filter} Depths')                               
    #depths_fig.show()
    #depths_fig.savefig(f'{save_path}/{survey}_{version}_{filter}_depths.png')
    
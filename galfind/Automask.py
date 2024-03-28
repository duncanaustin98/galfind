#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:51:51 2023

@author: austind
"""

# Automask.py

import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.visualization as vis
from kneed import KneeLocator
from photutils.aperture import CircularAperture
from scipy.ndimage import uniform_filter
from tqdm import tqdm as tq
from matplotlib import cm
from matplotlib.patches import Circle
from astropy.table import Table, Column
import cv2 as cv2
import os
from astropy.units import Quantity
from astropy.io import fits
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from scipy import ndimage
from skimage import morphology
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from astroquery.gaia import Gaia
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.visualization as vis
from scipy.ndimage.filters import convolve
import regions as reg
from scipy.fft import fft
from scipy.signal import find_peaks
#cmap = plt.cm.get_cmap('magma_r')
Gaia.ROW_LIMIT = 500

#os.environ['GALFIND_CONFIG_PATH'] = '/nvme/scratch/work/tharvey/GALFIND_WORK/galfind_config.ini'
from . import Data, Catalogue

def mask_image(image_path, seg_path='', image_ext=1, 
            seg_ext=0, write=True,
            edge_mask_distance = 50, mask_stars=True, scale_extra = 0.2, pixel_scale=0.03,
            mask_a=694.7, mask_b=3.5, exclude_gaia_galaxies=True):
    stretch = vis.CompositeStretch(vis.LogStretch(), vis.ContrastBiasStretch(contrast=30, bias=0.08))    
    norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)
    if write:
        mode = 'update'
    else:
        mode = 'readonly'
    fits_image = fits.open(image_path, mode=mode)
    data = fits_image[image_ext].data
    header = fits_image[image_ext].header
    wcs = WCS(header)

    scale_factor = scale_extra * np.array([data.shape[1], data.shape[0]])
    vertices_pix = [(-scale_factor[0], -scale_factor[1]), (-scale_factor[0], data.shape[0]+scale_factor[1]), (data.shape[1]+scale_factor[0], data.shape[0]+scale_factor[1]), (data.shape[1]+scale_factor[0], -scale_factor[1])]    
    print(vertices_pix)
    vertices_sky = wcs.all_pix2world(vertices_pix, 0)


    print('Opening seg map')
    if seg_path != '':
        seg = fits.open(seg_path)
        seg_data = seg[seg_ext].data
        seg_mask = seg_data > 0

    if mask_stars:
        gaia_stars = query_stars_in_polygon(vertices_sky)
        print(f'Found {len(gaia_stars)} stars in the region.')
        if exclude_gaia_galaxies:
            gaia_stars = gaia_stars[gaia_stars['vari_best_class_name'] != 'GALAXY']
            gaia_stars = gaia_stars[gaia_stars['classlabel_dsc_joint'] != 'galaxy']
            #gaia_stars = gaia_stars[gaia_stars['radius_sersic'] > 0]
            print(f'Found {len(gaia_stars)} stars in the region after excluding galaxies.')
            
        ra_gaia = np.asarray(gaia_stars['ra'])
        dec_gaia = np.asarray(gaia_stars['dec'])
        x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_stars['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_stars.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_stars.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_stars.add_column(Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111, projection=wcs)
    stretch = vis.CompositeStretch(vis.LogStretch(), vis.ContrastBiasStretch(contrast=30, bias=0.08))    
    norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)

    ax.imshow(data, cmap='Greys', origin='lower', interpolation='None', norm=norm)

    composite = lambda x_coord, y_coord, scale, angle: f'''# Region file format: DS9 version 4.1
                                                        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
                                                        image
                                                        composite({x_coord},{y_coord},{angle}) || composite=1
                                                            circle({x_coord},{y_coord},{163*scale}) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{730*scale},300.15) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{730*scale},240.00) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{730*scale},360.00) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{300*scale},269.48) ||'''

   
    diffraction_regions = []
    region_strings = []
    print('Masking stars')
    for pos, row in tq(enumerate(gaia_stars)):    
        # Plot circle
        ax.add_patch(Circle((row['x_pix'], row['y_pix']), 2*row['rmask_arcsec']/pixel_scale, color='r', fill=False, lw=2))
        scale = 2*row['rmask_arcsec']/pixel_scale / 730 
        sky_region = composite(row['x_pix'], row['y_pix'], scale, 0)
        region_obj = reg.Regions.parse(sky_region, format='ds9')
        diffraction_regions.append(region_obj)
        region_strings.append(region_obj.serialize(format='ds9'))

    mask_sky = np.zeros(data.shape, dtype=bool)
    print('Plotting masked stars')
    for pos1, regions in tq(enumerate(diffraction_regions)):
        for pos2, region in enumerate(regions):
            if pos1 == 0 and pos2 == 0:
                composite_region = region
            else:
                composite_region = composite_region | region
            idx_large, _ = region.to_mask(mode='center').get_overlap_slices(data.shape)
            mask_sky[idx_large] = True
            '''overlap = np.sum(data[idx_large], axis=None)
            if overlap == 0:
                region_strings.pop(pos1)
            else:'''
            artist = region.as_artist()
            ax.add_patch(artist)
    
    print('Done with composite region')
    return region_strings, mask_sky

def query_stars_in_polygon(vertices):
    '''
    Get Gaia stars in a polygon region defined by the vertices.

    Inputs:
    vertices: list of tuples, the vertices of the polygon in ICRS coordinates.
    '''
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = 500
    # Construct the ADQL query string
    adql_query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag, radius_sersic, classlabel_dsc_joint, vari_best_class_name
FROM gaiadr3.gaia_source 
LEFT OUTER JOIN gaiadr3.galaxy_candidates USING (source_id) 
WHERE 1 = CONTAINS(
    POINT('ICRS', ra, dec), 
    POLYGON('ICRS', 
        POINT('ICRS', {vertices[0][0]}, {vertices[0][1]}), 
        POINT('ICRS', {vertices[1][0]}, {vertices[1][1]}), 
        POINT('ICRS', {vertices[2][0]}, {vertices[2][1]}), 
        POINT('ICRS', {vertices[3][0]}, {vertices[3][1]}))) 
"""
    
    # Execute the query asynchronously
    job = Gaia.launch_job_async(adql_query)
    results = job.get_results()

    return results

def mask_edges(data, edge_value = 0, edge_mask_distance = 50, element='ELLIPSE'):
    import cv2

    fill= data==edge_value #true false array of where 0's are
    edges = fill*1 #convert to 1 for true and 0 for false
    edges = edges.astype(np.uint8) #dtype for cv2
    print('mask edges')
    if element == 'RECT':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_mask_distance,edge_mask_distance)) 
    elif element == 'ELLIPSE':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_mask_distance,edge_mask_distance))
    else:
        raise ValueError('Element must be RECT or ELLIPSE')

    dilate = cv2.dilate(edges, kernel, iterations=1) #dilate mask using the circle
    edges = 1 - dilate #invert mask, so it is 1 where it is not masked and 0 where it is masked
    #data = data * edges #apply mask (edited) 
    return edges

def make_mask(self, band, edge_mask_distance = 50, mask_stars=True, scale_extra = 0.2,
            mask_a=694.7, mask_b=3.5, exclude_gaia_galaxies=True, angle=0, edge_value = 0, 
            element='ELLIPSE'):
    '''
    This function will make a mask for the image based on the segmentation map, image edges and Gaia stars. 
    Requires astroquery to function.
    Only works for NIRCam images, and not for more complex PSFs like mosaics at different PA's. 
    Notes: mask_a, mask_b should probably be band dependent, but for now are fixed.

    Inputs:
    band: str, the band to make the mask for.
    edge_mask_distance: int, the distance in pixels to mask from the edge of the image.
    mask_stars: bool, whether to mask Gaia stars in the image.
    scale_extra: float, the factor to scale the image by to include diffraction spikes from stars outside the image footprint.
    mask_a: float, the overall scaling parameter for the mask size of the Gaia stars.
    mask_b: float, the scaling parameter for the exponential component of the mask size of the Gaia stars.
    angle: float, the angle of the diffraction spikes from the stars - default is 0 for diffraction spikes aligned with x-y axes.
    edge_value: int, the value of the image edge to mask.
    element: str, the shape of the element to use for masking the image edges. Options are 'RECT' or 'ELLIPSE'.
    Outputs:
    None
    '''
    # Composite region lambda function


    composite = lambda x_coord, y_coord, scale, angle: f'''# Region file format: DS9 version 4.1
                                                        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
                                                        image
                                                        composite({x_coord},{y_coord},{angle}) || composite=1
                                                            circle({x_coord},{y_coord},{163*scale}) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{730*scale},300.15) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{730*scale},240.00) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{730*scale},360.00) ||
                                                            ellipse({x_coord},{y_coord},{29*scale},{300*scale},269.48) ||'''

    # Load data
    im_data, im_header, seg_data, seg_header = self.load_data(band, incl_mask = False)
    pixel_scale = self.im_pixel_scales[band]
    wcs = WCS(im_header)
    # Scale up the image by boundary by scale_extra factor to include diffraction spikes from stars outside image footprint
    scale_factor = scale_extra * np.array([im_data.shape[1], im_data.shape[0]])
    vertices_pix = [(-scale_factor[0], -scale_factor[1]), (-scale_factor[0], data.shape[0]+scale_factor[1]), (data.shape[1]+scale_factor[0], data.shape[0]+scale_factor[1]), (data.shape[1]+scale_factor[0], -scale_factor[1])]    
    # Convert to sky coordinates
    vertices_sky = wcs.all_pix2world(vertices_pix, 0)

    if mask_stars:
        # Get list of Gaia stars in the polygon region
        gaia_stars = query_stars_in_polygon(vertices_sky)
        if exclude_gaia_galaxies:
            gaia_stars = gaia_stars[gaia_stars['vari_best_class_name'] != 'GALAXY']
            gaia_stars = gaia_stars[gaia_stars['classlabel_dsc_joint'] != 'galaxy']
            
        ra_gaia = np.asarray(gaia_stars['ra'])
        dec_gaia = np.asarray(gaia_stars['dec'])
        x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)
        
        # Generate mask scale for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_stars['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_stars.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_stars.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_stars.add_column(Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

    # Diagnositc plot
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111, projection=wcs)
    stretch = vis.CompositeStretch(vis.LogStretch(), vis.ContrastBiasStretch(contrast=30, bias=0.08))    
    norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)

    ax.imshow(data, cmap='Greys', origin='lower', interpolation='None', norm=norm)

    diffraction_regions = []
    region_strings = []
    print('Masking stars')
    for pos, row in tq(enumerate(gaia_stars)):    
        # Plot circle
        ax.add_patch(Circle((row['x_pix'], row['y_pix']), 2*row['rmask_arcsec']/pixel_scale, color='r', fill=False, lw=2))
        scale = 2*row['rmask_arcsec']/pixel_scale / 730 
        sky_region = composite(row['x_pix'], row['y_pix'], scale, angle)
        region_obj = reg.Regions.parse(sky_region, format='ds9')
        diffraction_regions.append(region_obj)
        region_strings.append(region_obj.serialize(format='ds9'))

    mask_stars = np.zeros(data.shape, dtype=bool)
    print('Plotting masked stars')
    for pos1, regions in tq(enumerate(diffraction_regions)):
        for pos2, region in enumerate(regions):
            if pos1 == 0 and pos2 == 0:
                composite_region = region
            else:
                composite_region = composite_region | region
            idx_large, _ = region.to_mask(mode='center').get_overlap_slices(data.shape)
            mask_stars[idx_large] = True
            artist = region.as_artist()
            ax.add_patch(artist)
    
    # Mask images edges

    mask = 1-mask_edges(data, edge_value = edge_value, edge_mask_distance = edge_mask_distance, element=element)
    # Mask up to 50 pixels from all edges - so edge is still masked if it as at edge of array
    mask[:edge_mask_distance, :] = mask[-edge_mask_distance:, :] = mask[:, :edge_mask_distance] = mask[:, -edge_mask_distance:] = 1
    
    mask = np.logical_or(mask.astype(np.uint8), mask_stars.astype(np.uint8))
    
    ax.imshow(mask, cmap='Reds', origin='lower', interpolation='None')

    # Don't need to mask objects for general mask, only for depths
    #seg_mask = seg_data > 0
    #mask = np.logical_or(mask, seg_mask)
    
    # Save mask - could save independent layers as well e.g. stars vs edges vs manual mask etc
    mask_hdu = fits.ImageHDU(mask.astype(np.uint8), header=wcs.to_header(), name='MASK')
    hdu = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
    hdu.writeto(f'{self.mask_dir}/{band}_basemask.fits', overwrite=True)

    # Save mask plot
    fig.savefig(f'{self.mask_dir}/{band}_mask.png')

    # Save ds9 region 
    with open(f'{self.mask_dir}/{band}_starmask.reg', 'w') as f:
        for region in region_strings:
            f.write(region + '\n')






# import numpy as np
# from astroquery.gaia import Gaia
# from astropy.coordinates import SkyCoord
# from astropy.io import fits
# from astropy import units as u
# from photutils import CircularAperture, RectangularAperture, detect_threshold, detect_sources
# from pyregion import parse, ShapeList

    # def mask_star(position, instrument, band, AB_mag):
#     # replicate the JWST/HST PSF
#     stellar_radius = 5. * u.arcsec # radius depends on the stellar AB mag only
    
#     centre = CircularAperture((position.ra.deg, position.dec.deg), r = (stellar_radius / instrument.pixel_scale).value)
#     #region_list = ShapeList([aperture.to_region() for aperture in apertures])
#     return centre
#     #pass
  
# def make_stellar_mask(band, im_data, im_header, instrument):
#     # get RA/DEC and extent of the image from the header
    
#     # Query GAIA DR3 to get star positions in a given region
#     print(band, im_header, im_header["RA_V1"], im_header["DEC_V1"])
#     coord = SkyCoord(ra = im_header["RA_V1"], dec = im_header["DEC_V1"], unit = (u.deg, u.deg), frame = 'icrs')  # Example coordinates for a JWST observation
#     x_ext = im_data.shape[1]
#     y_ext = im_data.shape[0]
#     print(x_ext, y_ext)
#     radius = np.max([im_data.shape[1], im_data.shape[0]]) * instrument.pixel_scale  # Example search radius
#     job = Gaia.cone_search_async(coord, radius)
#     gaia_table = job.get_results()

#     # Extract star positions from the GAIA table
#     ra = gaia_table['ra']
#     dec = gaia_table['dec']
#     positions = SkyCoord(ra = ra, dec = dec, unit = (u.deg, u.deg), frame = 'icrs')

#     # Create circular apertures around each star position
#     star_masks = np.array([mask_star(position, instrument, band, 0.) for position in positions])
#     stellar_mask = ShapeList([star_mask.to_region() for star_mask in star_masks])
#     return stellar_mask

# def make_edge_mask(im_data):
#     # Create rectangular apertures around the image edges
#     edge_apertures = [RectangularAperture(positions=[(0,0), (im_data.shape[1], 10)], w=0, h=im_data.shape[0]),
#                   RectangularAperture(positions=[(0,0), (10, im_data.shape[0])], w=im_data.shape[1], h=0),
#                   RectangularAperture(positions=[(0, im_data.shape[0]-10), (im_data.shape[1], im_data.shape[0])], w=0, h=im_data.shape[0]),
#                   RectangularAperture(positions=[(im_data.shape[1]-10, 0), (im_data.shape[1], im_data.shape[0])], w=im_data.shape[1], h=0)]

# def make_image_mask(im_data, stellar_mask, edge_mask):
#     # Combine the apertures into a single mask
#     all_apertures = stellar_mask + edge_mask
#     mask = create_mask(all_apertures, shape = im_data.shape)

#     # Apply the mask to the image
#     im_data_masked = im_data * mask
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:51:51 2023

@author: austind
"""

# image_masks.py
import numpy as np
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from photutils import CircularAperture, RectangularAperture, detect_threshold, detect_sources
from pyregion import parse, ShapeList

def mask_star(position, instrument, band, AB_mag):
    # replicate the JWST/HST PSF
    stellar_radius = 5. * u.arcsec # radius depends on the stellar AB mag only
    
    centre = CircularAperture((position.ra.deg, position.dec.deg), r = (stellar_radius / instrument.pixel_scale).value)
    
    #region_list = ShapeList([aperture.to_region() for aperture in apertures])
    return centre
    #pass
  
def make_stellar_mask(band, im_data, im_header, instrument):
    # get RA/DEC and extent of the image from the header
    
    # Query GAIA DR3 to get star positions in a given region
    print(band, im_header, im_header["RA_V1"], im_header["DEC_V1"])
    coord = SkyCoord(ra = im_header["RA_V1"], dec = im_header["DEC_V1"], unit = (u.deg, u.deg), frame = 'icrs')  # Example coordinates for a JWST observation
    x_ext = im_data.shape[1]
    y_ext = im_data.shape[0]
    print(x_ext, y_ext)
    radius = np.max([im_data.shape[1], im_data.shape[0]]) * instrument.pixel_scale  # Example search radius
    job = Gaia.cone_search_async(coord, radius)
    gaia_table = job.get_results()

    # Extract star positions from the GAIA table
    ra = gaia_table['ra']
    dec = gaia_table['dec']
    positions = SkyCoord(ra = ra, dec = dec, unit = (u.deg, u.deg), frame = 'icrs')

    # Create circular apertures around each star position
    star_masks = np.array([mask_star(position, instrument, band, 0.) for position in positions])
    stellar_mask = ShapeList([star_mask.to_region() for star_mask in star_masks])
    return stellar_mask

def make_edge_mask(im_data):
    # Create rectangular apertures around the image edges
    edge_apertures = [RectangularAperture(positions=[(0,0), (im_data.shape[1], 10)], w=0, h=im_data.shape[0]),
                  RectangularAperture(positions=[(0,0), (10, im_data.shape[0])], w=im_data.shape[1], h=0),
                  RectangularAperture(positions=[(0, im_data.shape[0]-10), (im_data.shape[1], im_data.shape[0])], w=0, h=im_data.shape[0]),
                  RectangularAperture(positions=[(im_data.shape[1]-10, 0), (im_data.shape[1], im_data.shape[0])], w=im_data.shape[1], h=0)]

def make_image_mask(im_data, stellar_mask, edge_mask):
    # Combine the apertures into a single mask
    all_apertures = stellar_mask + edge_mask
    mask = create_mask(all_apertures, shape = im_data.shape)

    # Apply the mask to the image
    im_data_masked = im_data * mask

if __name__ == "__main__":
    main()
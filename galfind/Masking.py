from __future__ import annotations

import astropy.units as u
import numpy as np
from regions import Regions
from astropy.io import fits
from astroquery.gaia import Gaia
from tqdm import tqdm
from astropy.table import Column
import cv2
import glob

from typing import Union, Optional, TYPE_CHECKING

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
if TYPE_CHECKING:
    from . import Band_Data, Band_Data_Base

from . import galfind_logger
from . import useful_funcs_austind as funcs

# Automasking

# # if "COSMOS-Web" in self.survey:
# #     # stellar masks the same for all bands
# #     star_mask_params = { # mask_a * exp(-mag / mask_b) is the form
# #         9000 * u.AA: {'mask_a': 700, 'mask_b': 3.7}}
# # else:
# #     star_mask_params = { # mask_a * exp(-mag / mask_b) is the form
# #         9000 * u.AA: {'mask_a': 1300, 'mask_b': 4},
# #         11500 * u.AA: {'mask_a': 1300, 'mask_b': 4},
# #         15000 * u.AA: {'mask_a': 1300, 'mask_b': 4},
# #         20000 * u.AA: {'mask_a': 1300, 'mask_b': 4},
# #         27700 * u.AA: {'mask_a': 1000, 'mask_b': 3.7},
# #         35600 * u.AA: {'mask_a': 800, 'mask_b': 3.7},
# #         44000 * u.AA: {'mask_a': 800, 'mask_b': 3.7},
# #     }

# # update to change scaling of central circle independently of spikes
# if star_mask_params is None:
#     star_mask_params_dict = {  # a * exp(-mag / b) in arcsec
#         11500 * u.AA: {
#             "central": {"a": 300.0, "b": 4.25},
#             "spikes": {"a": 400.0, "b": 4.5},
#         },
#     }
#     # Get closest wavelength parameters
#     closest_wavelength = min(
#         star_mask_params_dict.keys(),
#         key=lambda x: abs(x - self.filt.WavelengthCen),
#     )
#     star_mask_params = star_mask_params_dict[closest_wavelength]


def auto_mask(
    self: Band_Data,
    edge_mask_distance: Union[int, float] = 50,
    star_mask_params: Optional[dict] = None,
    scale_extra: float = 0.2,
    exclude_gaia_galaxies: bool = True,
    angle: float = -70.0,
    edge_value: float = 0.0,
    element: str = "ELLIPSE",
    gaia_row_lim: int = 500,
):
    _check_star_mask_params(star_mask_params)
    galfind_logger.info(f"Automasking {self.survey} {self.filt.band_name}.")

    if (
        "NIRCam" not in self.instr_name and star_mask_params is not None
    ):  # doesnt stop e.g. ACS_WFC+NIRCam from making star masks
        galfind_logger.critical(
            "Mask making only implemented for NIRCam data!"
        )
        raise (Exception("Star mask making only implemented for NIRCam data!"))

    # angle rotation is anti-clockwise for positive angles
    composite = (
        lambda x_coord,
        y_coord,
        central_scale,
        spike_scale,
        angle: f"""# Region file format: DS9 version 4.1
        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
        image
        composite({x_coord},{y_coord},0.00) || composite=1
            circle({x_coord},{y_coord},{163*central_scale}) ||
            ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{730*spike_scale},{str(np.round(300.15 + angle, 2))}) ||
            ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{730*spike_scale},{str(np.round(240. + angle, 2))}) ||
            ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{730*spike_scale},{str(np.round(360. + angle, 2))}) ||
            ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{300*spike_scale},{str(np.round(269.48 + angle, 2))}) ||"""
    )

    # Load data
    im_data, im_header, seg_data, seg_header = self.load_data(incl_mask=False)
    wcs = self.load_wcs()

    # Scale up the image by boundary by scale_extra factor to include diffraction spikes from stars outside image footprint
    scale_factor = scale_extra * np.array([im_data.shape[1], im_data.shape[0]])
    vertices_pix = [
        (-scale_factor[0], -scale_factor[1]),
        (-scale_factor[0], im_data.shape[0] + scale_factor[1]),
        (
            im_data.shape[1] + scale_factor[0],
            im_data.shape[0] + scale_factor[1],
        ),
        (im_data.shape[1] + scale_factor[0], -scale_factor[1]),
    ]
    # Convert to sky coordinates
    vertices_sky = wcs.all_pix2world(vertices_pix, 0)

    # Diagnostic plot
    # if plot:
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection=wcs)
    #     stretch = vis.CompositeStretch(
    #         vis.LogStretch(),
    #         vis.ContrastBiasStretch(contrast=30, bias=0.08),
    #     )
    #     norm = ImageNormalize(stretch=stretch, vmin=0.001, vmax=10)

    #     ax.imshow(
    #         im_data,
    #         cmap="Greys",
    #         origin="lower",
    #         interpolation="None",
    #         norm=norm,
    #     )

    if star_mask_params is not None:
        galfind_logger.debug(
            f"Making stellar mask for {self.survey} {self.version} {self.filt.band_name}"
        )
        # Get list of Gaia stars in the polygon region
        Gaia.ROW_LIMIT = gaia_row_lim
        # Construct the ADQL query string
        adql_query = f"""
            SELECT source_id, ra, dec, phot_g_mean_mag, radius_sersic, classlabel_dsc_joint, vari_best_class_name
            FROM gaiadr3.gaia_source 
            LEFT OUTER JOIN gaiadr3.galaxy_candidates USING (source_id) 
            WHERE 1 = CONTAINS(
                POINT('ICRS', ra, dec), 
                POLYGON('ICRS', 
                    POINT('ICRS', {vertices_sky[0][0]}, {vertices_sky[0][1]}), 
                    POINT('ICRS', {vertices_sky[1][0]}, {vertices_sky[1][1]}), 
                    POINT('ICRS', {vertices_sky[2][0]}, {vertices_sky[2][1]}), 
                    POINT('ICRS', {vertices_sky[3][0]}, {vertices_sky[3][1]})))"""

        # Execute the query asynchronously
        job = Gaia.launch_job_async(adql_query)
        gaia_stars = job.get_results()
        print(f"Found {len(gaia_stars)} stars in the region.")
        if exclude_gaia_galaxies:
            gaia_stars = gaia_stars[
                gaia_stars["vari_best_class_name"] != "GALAXY"
            ]
            gaia_stars = gaia_stars[
                gaia_stars["classlabel_dsc_joint"] != "galaxy"
            ]
            # Remove masked flux values
            gaia_stars = gaia_stars[~np.isnan(gaia_stars["phot_g_mean_mag"])]

        ra_gaia = np.asarray(gaia_stars["ra"])
        dec_gaia = np.asarray(gaia_stars["dec"])
        x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)

        # Generate mask scale for each star
        central_scale_stars = (
            2.0
            * star_mask_params["central"]["a"]
            / (730.0 * self.im_pixel_scale.to(u.arcsec).value)
        ) * np.exp(
            -gaia_stars["phot_g_mean_mag"] / star_mask_params["central"]["b"]
        )
        spike_scale_stars = (
            2.0
            * star_mask_params["spikes"]["a"]
            / (730.0 * self.im_pixel_scale.to(u.arcsec).value)
        ) * np.exp(
            -gaia_stars["phot_g_mean_mag"] / star_mask_params["spikes"]["b"]
        )
        # Update the catalog
        gaia_stars.add_column(Column(data=x_gaia, name="x_pix"))
        gaia_stars.add_column(Column(data=y_gaia, name="y_pix"))

        diffraction_regions = []
        region_strings = []
        for pos, (row, central_scale, spike_scale) in tqdm(
            enumerate(zip(gaia_stars, central_scale_stars, spike_scale_stars))
        ):
            # Plot circle
            # if plot:
            #     ax.add_patch(Circle((row['x_pix'], row['y_pix']), 2 * row['rmask_arcsec'] / pixel_scale, color = 'r', fill = False, lw = 2))
            sky_region = composite(
                row["x_pix"],
                row["y_pix"],
                central_scale,
                spike_scale,
                angle,
            )
            region_obj = Regions.parse(sky_region, format="ds9")
            diffraction_regions.append(region_obj)
            region_strings.append(region_obj.serialize(format="ds9"))

        stellar_mask = np.zeros(im_data.shape, dtype=bool)
        for regions in tqdm(diffraction_regions):
            for region in regions:
                idx_large, idx_little = region.to_mask(
                    mode="center"
                ).get_overlap_slices(im_data.shape)
                # idx_large is x,y box containing bounds of region in image
                if idx_large is not None:
                    stellar_mask[idx_large] = np.logical_or(
                        region.to_mask().data[idx_little],
                        stellar_mask[idx_large],
                    )
                # if plot:
                #     artist = region.as_artist()
                #     ax.add_patch(artist)

    # Mask image edges
    fill = np.logical_or(
        (im_data == edge_value), np.isnan(im_data)
    )  # true false array of where 0's are
    # also fill in nans
    edges = fill * 1  # convert to 1 for true and 0 for false
    edges = edges.astype(np.uint8)  # dtype for cv2
    galfind_logger.debug(
        f"Masking edges for {self.survey} {self.filt.band_name}."
    )
    if element == "RECT":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (edge_mask_distance, edge_mask_distance)
        )
    elif element == "ELLIPSE":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (edge_mask_distance, edge_mask_distance)
        )
    else:
        raise ValueError(f"element = {element} must be 'RECT' or 'ELLIPSE'")

    edge_mask = cv2.dilate(
        edges, kernel, iterations=1
    )  # dilate mask using the circle

    # Mask up to 50 pixels from all edges - so edge is still masked if it as at edge of array
    edge_mask[:edge_mask_distance, :] = edge_mask[
        -edge_mask_distance:, :
    ] = edge_mask[:, :edge_mask_distance] = edge_mask[
        :, -edge_mask_distance:
    ] = 1

    if star_mask_params is not None:
        full_mask = np.logical_or(
            edge_mask.astype(np.uint8), stellar_mask.astype(np.uint8)
        )
    else:
        full_mask = edge_mask.astype(np.uint8)

    # if plot:
    #     ax.imshow(
    #         full_mask, cmap="Reds", origin="lower", interpolation="None"
    #     )
    # if plot:
    #     # Save mask plot
    #     fig.savefig(f"{self.mask_dir}/{self.filt.band_name}_mask.png", dpi=300)
    #     funcs.change_file_permissions(f"{self.mask_dir}/{self.filt.band_name}_mask.png")

    # Check for artifacts mask to combine with exisitng mask
    files = glob.glob(f"{self.mask_dir}/{self.filt.band_name}*.reg")
    # Check for 'artifact' in file name
    files = [file for file in files if "artifact" in file]
    artifact_mask = None
    if len(files) > 0:
        artifact_mask = np.zeros(im_data.shape, dtype=bool)
        galfind_logger.debug(f"Found {len(files)} artifact masks")
        for file in files:
            galfind_logger.debug(f"Adding mask {file}")
            mask = Regions.read(file)

            for region in mask:
                region = region.to_pixel(wcs)
                idx_large, idx_little = region.to_mask(
                    mode="center"
                ).get_overlap_slices(im_data.shape)
                if idx_large is not None:
                    full_mask[idx_large] = np.logical_or(
                        region.to_mask().data[idx_little],
                        full_mask[idx_large],
                    )
                    artifact_mask[idx_large] = np.logical_or(
                        region.to_mask().data[idx_little],
                        artifact_mask[idx_large],
                    )

    # Save mask - could save independent layers as well e.g. stars vs edges vs manual mask etc
    output_mask_path = (
        f"{self.mask_dir}/fits_masks/{self.filt.band_name}_basemask.fits"
    )

    funcs.make_dirs(output_mask_path)
    full_mask_hdu = fits.ImageHDU(
        full_mask.astype(np.uint8), header=wcs.to_header(), name="MASK"
    )
    edge_mask_hdu = fits.ImageHDU(
        edge_mask.astype(np.uint8), header=wcs.to_header(), name="EDGE"
    )
    hdulist = [fits.PrimaryHDU(), full_mask_hdu, edge_mask_hdu]
    if star_mask_params is not None:
        stellar_mask_hdu = fits.ImageHDU(
            stellar_mask.astype(np.uint8),
            header=wcs.to_header(),
            name="STELLAR",
        )
        hdulist.append(stellar_mask_hdu)
    if artifact_mask is not None:
        artifact_mask_hdu = fits.ImageHDU(
            artifact_mask.astype(np.uint8),
            header=wcs.to_header(),
            name="ARTIFACT",
        )
        hdulist.append(artifact_mask_hdu)

    hdu = fits.HDUList(hdulist)
    hdu.writeto(output_mask_path, overwrite=True)
    # Change permission to read/write for all
    funcs.change_file_permissions(output_mask_path)

    # Save ds9 region
    starmask_path = f"{self.mask_dir}/{self.filt.band_name}_starmask.reg"
    funcs.make_dirs(starmask_path)
    if star_mask_params is not None:
        with open(starmask_path, "w") as f:
            for region in region_strings:
                f.write(region + "\n")
        funcs.change_file_permissions(starmask_path)
    return output_mask_path


def _check_star_mask_params(star_mask_params):
    assert isinstance(star_mask_params, dict), galfind_logger.warning(
        f"Mask overridden, but {type(star_mask_params)=} != dict"
    )
    assert (
        "central" in star_mask_params.keys()
        and "spikes" in star_mask_params.keys()
    )
    assert type(star_mask_params["central"]) == dict, galfind_logger.warning(
        f"Mask overridden, but {type(star_mask_params['central'])=} != dict"
    )
    assert (
        "a" in star_mask_params["central"].keys()
        and "b" in star_mask_params["central"].keys()
    )
    assert type(star_mask_params["spikes"]) == dict, galfind_logger.warning(
        f"Mask overridden, but {type(star_mask_params['spikes'])=} != dict"
    )
    assert (
        "a" in star_mask_params["spikes"].keys()
        and "b" in star_mask_params["spikes"].keys()
    )
    assert all(
        type(scale) in [float, int]
        for mask_type in star_mask_params.values()
        for scale in mask_type.values()
    )

from __future__ import annotations

import astropy.units as u
import numpy as np
from regions import Regions
from astropy.io import fits
from astroquery.gaia import Gaia
from tqdm import tqdm
from astropy.table import Column
import cv2
from pathlib import Path
import os
import glob

from typing import List, Dict, Tuple, Union, Optional, NoReturn, TYPE_CHECKING

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
if TYPE_CHECKING:
    from . import Band_Data_Base, Stacked_Band_Data, Filter

from . import config, galfind_logger
from . import useful_funcs_austind as funcs

# Manual masking

auto_mask_keys = [
    "METHOD",
    "CENTRAL_A",
    "CENTRAL_B",
    "SPIKES_A",
    "SPIKES_B",
    "EDGE_DIST",
    "SCALE_EXTRA",
    "EXCLUDE_GAIA",
    "ANGLE",
    "EDGE_VALUE",
    "ELEMENT",
    "GAIA_ROW_LIM",
]

def get_mask_args(
    fits_mask_path: str,
) -> Tuple[
    Union[None, str], Dict[str, Union[str, int, float, Dict[str, float]]]
]:
    # return None if fits_mask_path is not a file
    if not Path(fits_mask_path).is_file():
        galfind_logger.warning(f"{fits_mask_path=} not a file")
        return None
    elif ".fits" not in fits_mask_path:
        galfind_logger.warning(f"{fits_mask_path=} not a .fits file")
        return None
    else:
        # open fits mask
        mask_hdr = fits.open(fits_mask_path, mode="readonly")[1].header
        if all(
            key in mask_hdr.keys()
            for key in auto_mask_keys
        ):
            mask_method = str(mask_hdr["METHOD"]).lower()
            assert mask_method.lower() == "auto", galfind_logger.critical(
                f"Mask method in {fits_mask_path} is {mask_method.lower()} not 'auto'"
            )
            # re-create star_mask_params dict
            star_mask_params = {
                "central": {
                    "a": mask_hdr["CENTRAL_A"],
                    "b": mask_hdr["CENTRAL_B"],
                },
                "spikes": {
                    "a": mask_hdr["SPIKES_A"],
                    "b": mask_hdr["SPIKES_B"],
                },
            }
            mask_args = {
                "method": mask_method,
                "star_mask_params": star_mask_params,
                "edge_mask_distance": mask_hdr["EDGE_DIST"],
                "scale_extra": mask_hdr["SCALE_EXTRA"],
                "exclude_gaia_galaxies": mask_hdr["EXCLUDE_GAIA"],
                "angle": mask_hdr["ANGLE"],
                "edge_value": mask_hdr["EDGE_VALUE"],
                "element": mask_hdr["ELEMENT"],
                "gaia_row_lim": mask_hdr["GAIA_ROW_LIM"],
            }
        else:
            mask_args = {"method": "manual"}
        return mask_args

def get_mask_method(fits_mask_path: str) -> str:
    mask_args = get_mask_args(fits_mask_path)
    if mask_args is not None:
        return mask_args["method"]
    else:
        return "manual"

def manually_mask(
    self: Type[Band_Data_Base],
    overwrite: bool = False,
) -> Union[None, NoReturn]:
    fits_mask_path = get_manual_fits_mask_path(self)
    if not Path(fits_mask_path).is_file() or overwrite:
        # if no fits mask found, search for region mask
        reg_mask_dir = f"{config['Masking']['MASK_DIR']}/{self.survey}/reg"
        if self.__class__.__name__ == "Stacked_Band_Data":
            reg_mask_dir += "/stacked"
        reg_mask_paths = []
        for filt_ext in [
            self.filt_name.upper(),
            self.filt_name.lower(),
            self.filt_name.lower().replace("f", "F"),
            self.filt_name.upper().replace("F", "f"),
        ]:
            reg_mask_paths.extend(
                list(glob.glob(f"{reg_mask_dir}/*{filt_ext}*.reg"))
            )
        assert len(reg_mask_paths) <= 1, galfind_logger.critical(
            f"{len(reg_mask_paths)=} > 1"
        )
        if len(reg_mask_paths) == 1:
            reg_mask_path = reg_mask_paths[0]
        else:
            raise (
                Exception(
                    "Neither .fits or .reg mask path found "
                    + f"for {self.survey} {self.version} {self.filt_name}"
                )
            )
        # clean region mask of any zero size regions
        # if not "clean" in the .reg filename
        if not "_clean" in reg_mask_path:
            # update reg_mask_path to cleaned version
            reg_mask_path = clean_reg_mask(reg_mask_path)
        # convert to fits mask
        fits_mask_path = convert_mask_to_fits(
            self, reg_mask_path, out_path=fits_mask_path
        )
    return fits_mask_path


def clean_reg_mask(mask_path: str) -> str:
    # open region file
    with open(mask_path, "r") as f:
        lines = f.readlines()
        clean_mask_path = mask_path.replace(".reg", "_clean.reg")
        with open(clean_mask_path, "w") as temp:
            for i, line in enumerate(lines):
                # if line.startswith('physical'):
                #     lines[i] = line.replace('physical', 'image')
                if i <= 2:
                    temp.write(line)
                if not (line.endswith(",0)\n") and line.startswith("circle")):
                    if (
                        (
                            line.startswith("ellipse")
                            and not (line.split(",")[2] == "0")
                            and not (line.split(",")[3] == "0")
                        )
                        or line.startswith("box")
                        or line.startswith("circle")
                        or line.startswith("polygon")
                    ):
                        temp.write(line)
    funcs.change_file_permissions(mask_path)
    funcs.change_file_permissions(clean_mask_path)
    # insert original mask ds9 region file into an unclean folder
    unclean_path = f"{funcs.split_dir_name(mask_path,'dir')}/unclean/{funcs.split_dir_name(mask_path,'name')}"
    funcs.make_dirs(unclean_path)
    os.rename(mask_path, unclean_path)
    return clean_mask_path


def convert_mask_to_fits(
    self: Type[Band_Data_Base], mask_path: str, out_path: Optional[str]
) -> Union[str, np.ndarray]:
    if out_path is None:
        convert = True
    elif not Path(out_path).is_file():
        convert = True
    else:
        convert = False
    if convert:
        # open image corresponding to band
        im_data = self.load_im()[0]
        # open .reg mask file
        mask_regions = Regions.read(mask_path)
        wcs = self.load_wcs()

        pix_mask = np.zeros(im_data.shape, dtype=bool)
        for region in mask_regions:
            if "Region" not in region.__class__.__name__:
                region = region.to_pixel(wcs)
            idx_large, idx_little = region.to_mask(
                mode="center"
            ).get_overlap_slices(im_data.shape)
            if idx_large is not None:
                pix_mask[idx_large] = np.logical_or(
                    region.to_mask().data[idx_little], pix_mask[idx_large]
                )
        if not out_path is None:
            hdr = wcs.to_header()
            hdr["METHOD"] = "manual"
            # make .fits mask
            mask_hdu = fits.ImageHDU(
                pix_mask.astype(np.uint8), header=hdr, name="MASK"
            )
            hdu = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
            hdu.writeto(out_path, overwrite=True)
            funcs.change_file_permissions(out_path)
            galfind_logger.info(
                f"Created fits mask from manually created reg mask, saving as {out_path}"
            )
        else:
            return pix_mask
    else:
        galfind_logger.info(
            f"fits mask at {out_path} already exists, skipping!"
        )
    return out_path


def get_manual_fits_mask_path(self: Type[Band_Data_Base]) -> str:
    fits_mask_dir = f"{config['Masking']['MASK_DIR']}/{self.survey}/manual"
    fits_mask_path = (
        f"{fits_mask_dir}/{self.filt_name}_{self.version}_manual.fits"
    )
    funcs.make_dirs(fits_mask_path)
    return fits_mask_path


def auto_mask(
    self: Type[Band_Data_Base],
    star_mask_params: Optional[dict] = None,
    edge_mask_distance: Union[int, float] = 50,
    scale_extra: float = 0.2,
    exclude_gaia_galaxies: bool = True,
    angle: float = -0.0,
    edge_value: float = 0.0,
    element: str = "ELLIPSE",
    gaia_row_lim: int = 500,
    overwrite: bool = False,
):
    output_mask_path = f"{config['Masking']['MASK_DIR']}/{self.survey}/auto/{self.filt_name}_auto.fits"
    funcs.make_dirs(output_mask_path)

    #print("auto_mask:", self, star_mask_params)

    if not Path(output_mask_path).is_file() or overwrite:
        check_star_mask_params(star_mask_params)
        galfind_logger.info(f"Automasking {self.survey} {self.filt_name}")

        if (
            "NIRCam" not in self.instr_name and star_mask_params is not None
        ):  
            star_mask_params = None
            # doesnt stop e.g. ACS_WFC+NIRCam from making star masks
            galfind_logger.warning(
                f"Stellar mask wanted for {self.filt_name} only implemented for NIRCam data!"
            )
            # raise (
            #     Exception("Star mask making only implemented for NIRCam data!")
            # )

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
        im_data = self.load_im()[0]
        wcs = self.load_wcs()

        # Scale up the image by boundary by scale_extra factor to include diffraction spikes from stars outside image footprint
        scale_factor = scale_extra * np.array(
            [im_data.shape[1], im_data.shape[0]]
        )
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

        reg_mask_dir = f"{config['Masking']['MASK_DIR']}/{self.survey}/reg"
        if star_mask_params is not None:
            galfind_logger.debug(
                f"Making stellar mask for {self.survey} {self.version} {self.filt_name}"
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
            #print(f"Found {len(gaia_stars)} stars in the region.")
            if exclude_gaia_galaxies:
                gaia_stars = gaia_stars[
                    gaia_stars["vari_best_class_name"] != "GALAXY"
                ]
                gaia_stars = gaia_stars[
                    gaia_stars["classlabel_dsc_joint"] != "galaxy"
                ]
                # Remove masked flux values
                gaia_stars = gaia_stars[
                    ~np.isnan(gaia_stars["phot_g_mean_mag"])
                ]

            ra_gaia = np.asarray(gaia_stars["ra"])
            dec_gaia = np.asarray(gaia_stars["dec"])
            x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)

            # Generate mask scale for each star
            central_scale_stars = (
                2.0
                * star_mask_params["central"]["a"]
                / (730.0 * self.pix_scale.to(u.arcsec).value)
            ) * np.exp(
                -gaia_stars["phot_g_mean_mag"]
                / star_mask_params["central"]["b"]
            )
            spike_scale_stars = (
                2.0
                * star_mask_params["spikes"]["a"]
                / (730.0 * self.pix_scale.to(u.arcsec).value)
            ) * np.exp(
                -gaia_stars["phot_g_mean_mag"]
                / star_mask_params["spikes"]["b"]
            )
            # Update the catalog
            gaia_stars.add_column(Column(data=x_gaia, name="x_pix"))
            gaia_stars.add_column(Column(data=y_gaia, name="y_pix"))

            diffraction_regions = []
            stellar_region_strings = []
            for pos, (row, central_scale, spike_scale) in tqdm(
                enumerate(
                    zip(gaia_stars, central_scale_stars, spike_scale_stars)
                )
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
                stellar_region_strings.append(
                    region_obj.serialize(format="ds9")
                )

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
            f"Masking edges for {self.survey} {self.filt_name}."
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
            raise ValueError(
                f"element = {element} must be 'RECT' or 'ELLIPSE'"
            )

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
            # Save ds9 region for stars
            starmask_path = (
                f"{reg_mask_dir}/stellar/{self.filt_name}_stellar.reg"
            )
            funcs.make_dirs(starmask_path)
            with open(starmask_path, "w") as f:
                for region in stellar_region_strings:
                    f.write(region + "\n")
                f.close()
            funcs.change_file_permissions(starmask_path)
        else:
            full_mask = edge_mask.astype(np.uint8)

        # Check for artefact masks to combine with exisitng mask
        artefact_mask_paths = []
        for filt_ext in [
            self.filt_name.upper(),
            self.filt_name.lower(),
            self.filt_name.lower().replace("f", "F"),
            self.filt_name.upper().replace("F", "f"),
        ]:
            artefact_mask_paths.extend(
                list(glob.glob(f"{reg_mask_dir}/artefact/*/*{filt_ext}*.reg"))
            )
        # make a dictionary of existing paths and their
        # corresponding (capitalized) host directories
        artefact_mask_dir_names = np.unique(
            [path.split("/")[-2].upper() for path in artefact_mask_paths]
        )
        artefact_mask_dict = {
            name: [
                path
                for path in artefact_mask_paths
                if path.split("/")[-2].upper() == name
            ]
            for name in artefact_mask_dir_names
        }

        assert not any(
            key in ["MASK", "EDGE"] for key in artefact_mask_dir_names
        ), galfind_logger.critical(
            f"{artefact_mask_dir_names=} cannot contain any of ['MASK', 'EDGE']"
        )
        if star_mask_params is None:
            assert not any(
                key in ["STELLAR"] for key in artefact_mask_dir_names
            ), galfind_logger.critical(
                f"{artefact_mask_dir_names=} cannot contain 'STELLAR'"
            )
        # make pixel masks from the paths
        artefact_pix_masks = {}
        for ext_name, mask_paths in artefact_mask_dict.items():
            #artefact_mask = np.zeros(im_data.shape, dtype=bool)
            galfind_logger.debug(f"Found {len(mask_paths)} {ext_name} masks")
            for i, path in enumerate(mask_paths):
                galfind_logger.debug(
                    f"Adding mask {path} to {ext_name} for {self.survey} {self.version} {self.filt_name}"
                )
                # clean path if not already done so
                if "_clean" not in path:
                    path = clean_reg_mask(path)
                # construct pixel mask
                pix_mask = convert_mask_to_fits(self, path, out_path=None)
                if i == 0:
                    artefact_pix_masks[ext_name] = [pix_mask]
                else:
                    artefact_pix_masks[ext_name].extend([pix_mask])
            # combine masks for each extension
            artefact_pix_masks[ext_name] = np.logical_or.reduce(
                tuple([mask.astype(np.uint8) for mask in artefact_pix_masks[ext_name]])
            )
        # update full mask to include all artefacts
        full_mask = np.logical_or(
            full_mask.astype(np.uint8),
            np.logical_or.reduce(
                tuple(
                    [
                        pix_mask.astype(np.uint8)
                        for pix_mask in artefact_pix_masks.values()
                    ]
                )
            ),
        )
        # make header
        hdr = wcs.to_header()
        hdr_args = {
            "METHOD": "auto", 
            "EDGE_DIST": edge_mask_distance,
            "SCALE_EXTRA": scale_extra,
            "EXCLUDE_GAIA": exclude_gaia_galaxies,
            "ANGLE": angle,
            "EDGE_VALUE": edge_value,
            "ELEMENT": element,
            "GAIA_ROW_LIM": gaia_row_lim,
        }
        if star_mask_params is not None:
            # add star mask parameters to header arguments
            stellar_mask_hdr_args = {
                "CENTRAL_A": star_mask_params["central"]["a"], 
                "CENTRAL_B": star_mask_params["central"]["b"],
                "SPIKES_A": star_mask_params["spikes"]["a"],
                "SPIKES_B": star_mask_params["spikes"]["b"],
            }
            hdr_args = {**hdr_args, **stellar_mask_hdr_args}
        # else:
        #     stellar_mask_hdr_args = {
        #         "CENTRAL_A": None, 
        #         "CENTRAL_B": None, 
        #         "SPIKES_A": None, 
        #         "SPIKES_B": None
        #     }

        for key, value in hdr_args.items():
            hdr[key] = value
        # Save mask
        full_mask_hdu = fits.ImageHDU(
            full_mask.astype(np.uint8), header=hdr, name="MASK"
        )
        edge_mask_hdu = fits.ImageHDU(
            edge_mask.astype(np.uint8), header=hdr, name="EDGE"
        )
        hdulist = [fits.PrimaryHDU(), full_mask_hdu, edge_mask_hdu]
        if star_mask_params is not None:
            stellar_mask_hdu = fits.ImageHDU(
                stellar_mask.astype(np.uint8),
                header=hdr,
                name="STELLAR",
            )
            hdulist.append(stellar_mask_hdu)
        for artefact_ext_name, artefact_pix_mask in artefact_pix_masks.items():
            artefact_mask_hdu = fits.ImageHDU(
                artefact_pix_mask.astype(np.uint8),
                header=hdr,
                name=artefact_ext_name,
            )
            hdulist.append(artefact_mask_hdu)

        hdu = fits.HDUList(hdulist)
        hdu.writeto(output_mask_path, overwrite=True)
        # Change permission to read/write for all
        funcs.change_file_permissions(output_mask_path)

    return output_mask_path, get_mask_args(output_mask_path)


def check_star_mask_params(
    star_mask_params: Dict[u.Quantity, Dict[str, float]],
) -> NoReturn:
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


def sort_band_dependent_star_mask_params(
    filt: Filter,
    star_mask_params: Optional[
        Union[
            Dict[str, Dict[str, float]],
            Dict[u.Quantity, Dict[str, Dict[str, float]]],
        ]
    ],
) -> Optional[Dict[str, Dict[str, float]]]:
    if not all(isinstance(key, str) for key in star_mask_params.keys()):
        # get closest wavelength to the filter in question
        closest_wavelength = min(
            star_mask_params.keys(),
            key=lambda x: abs(x - filt.WavelengthCen),
        )
        star_mask_params = star_mask_params[closest_wavelength]
    return star_mask_params


def get_combined_path_name(self: Stacked_Band_Data) -> str:
    out_dir = f"{config['Masking']['MASK_DIR']}/{self.survey}/combined"
    if all(band_data.mask_args["method"] == self.band_data_arr[0].mask_args["method"] for band_data in self.band_data_arr):
        filt_name_mask_method = '+'.join([band_data.filt_name for band_data in self.band_data_arr]) + \
            f"_{self.band_data_arr[0].mask_args['method']}"
    else:
        filt_name_mask_method = "+".join(
            [
                f"{band_data.filt_name}-{band_data.mask_args['method']}"
                for band_data in self.band_data_arr
            ]
        )
    out_name = f"{self.survey}_{filt_name_mask_method}.fits"
    out_path = f"{out_dir}/{out_name}"
    funcs.make_dirs(out_path)
    return out_path

def combine_masks(self: Stacked_Band_Data) -> str:
    out_path = get_combined_path_name(self)
    if not Path(out_path).is_file():
        assert all(
            band_data.pix_scale == self.band_data_arr[0].pix_scale
            for band_data in self.band_data_arr
        ), galfind_logger.critical("All bands must have the same pixel scale")
        assert all(
            band_data.data_shape == self.band_data_arr[0].data_shape
            for band_data in self.band_data_arr
        ), galfind_logger.critical("All bands must have the same data shape")
        band_mask_exts = [
            band_data.load_mask()[0] for band_data in self.band_data_arr
        ]
        all_exts = list(
            np.unique([list(mask_ext.keys()) for mask_ext in band_mask_exts])
        )
        assert all(
            "MASK" in band_mask_ext.keys() for band_mask_ext in band_mask_exts
        ), galfind_logger.critical("All bands must have a 'MASK' extension")
        # load wcs from the reddest band
        hdr = self.band_data_arr[-1].load_mask()[1]["MASK"]
        for key in auto_mask_keys:
            if key in hdr.keys():
                hdr.remove(key)
        hdr_dict = {band_data.filt_name: band_data.load_mask()[1]["MASK"] for band_data in self.band_data_arr}
        for band_name, band_hdr in hdr_dict.items():
            for key, value in band_hdr.items():
                if key in auto_mask_keys:
                    hdr[f"{key}_{band_name}"] = value
        print(list(dict(hdr).keys()))
        # combine masks for each valid extension contained in all masks
        combined_mask_hdul = [fits.PrimaryHDU()]
        for ext in all_exts:
            band_masks = [
                band_mask_ext[ext]
                for band_mask_ext in band_mask_exts
                if ext in band_mask_ext.keys()
            ]
            assert (
                mask.shape == band_masks[0].shape for mask in band_masks
            ), galfind_logger.critical("All masks must have the same shape")
            combined_mask = np.logical_or.reduce(tuple(band_masks))
            combined_mask_hdul.extend([
                fits.ImageHDU(
                    combined_mask.astype(np.uint8),
                    header=hdr,
                    name=ext,
                )
            ])
        hdul = fits.HDUList(combined_mask_hdul)
        hdul.writeto(out_path, overwrite=True)
        funcs.change_file_permissions(out_path)
        galfind_logger.info(f"Created combined mask for {repr(self)}")
    else:
        galfind_logger.info(
            f"Combined mask for {repr(self)} already exists at {out_path}"
        )
    mask_args = {band_data.filt_name: band_data.mask_args for band_data in self.band_data_arr}
    return out_path, mask_args


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


# def plot_mask_from_data(self, ax, label, show=True):
#     mask = self.load_mask(incl_mask=True)[4]
#     cbar_in = ax.imshow(mask, origin="lower")
#     plt.title(f"{label} mask")
#     plt.xlabel("X / pix")
#     plt.ylabel("Y / pix")
#     plt.colorbar(cbar_in)
#     if show:
#         plt.show()

# def plot_mask_regions_from_band(self, ax):
#     mask_regions = Regions.read(self.mask_path)
#     patch_list = [reg.as_artist() for reg in mask_regions]
#     for p in patch_list:
#         ax.add_patch(p)

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

# if plot:
#     ax.imshow(
#         full_mask, cmap="Reds", origin="lower", interpolation="None"
#     )
# if plot:
#     # Save mask plot
#     fig.savefig(f"{self.mask_dir}/{self.filt.band_name}_mask.png", dpi=300)
#     funcs.change_file_permissions(f"{self.mask_dir}/{self.filt.band_name}_mask.png")

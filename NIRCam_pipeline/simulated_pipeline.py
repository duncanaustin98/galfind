#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:36:34 2023

@author: austind
"""

# NIRCam_pipeline.py

import astropy.units as u

from galfind import (
    Catalogue,
)
from galfind.Catalogue_Creator import Galfind_Catalogue_Creator


def simulated_pipeline(
    survey,
    fits_cat_path,
    version,
    instruments,
    aper_diams,
    code_names,
    eazy_zmax_lowz,
    min_flux_pc_errs,
    excl_bands,
    cat_type="loc_depth",
    eazy_templates="fsps_larson",
    zero_point=31.4,
):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = Galfind_Catalogue_Creator(
            cat_type, aper_diams[0], pc_err, zero_point=zero_point
        )
        cat = Catalogue.from_fits_cat(
            fits_cat_path,
            version,
            instruments,
            cat_creator,
            code_names,
            survey,
            eazy_zmax_lowz,
            templates_arr=eazy_templates,
            data=None,
            mask=False,
            excl_bands=excl_bands,
        )
        # for i, code in enumerate(sed_codes):
        #     cat = code.fit_cat(cat, templates = eazy_templates)
        #     #code.fit_cat(cat, templates = eazy_templates)
        #     # calculate the extended source corrections
        #     if code.code_name == "LePhare":
        #         cat.make_ext_src_corr_cat(code.code_name)
        #     # calculate the UV properties for this catalogue
        #     if instruments == ["NIRCam"]: # QUICK FIX!
        #         print("Instruments name is a QUICK FIX!")
        #         instruments_name = "NIRCam"
        #     cat.make_UV_fit_cat(UV_PDF_path = f"{config['RestUVProperties']['UV_PDF_PATH']}/{version}/{instruments_name}/{survey}/{code.code_name}+{pc_err}pc")


if __name__ == "__main__":
    version = "simulated"
    fits_cat_path = "/raid/scratch/data/JAGUAR/JAGUAR_SimDepth_CEERS_Cut.fits"
    fits_cat_path = "/nvme/scratch/work/tharvey/catalogs/JAGUAR_SimDepth_NGDEEP_z=9_10.fits"
    fits_cat_path = "/raid/scratch/data/JAGUAR/JAGUAR_SimDepth_ELG.fits"

    # fits_cat_path = '/nvme/scratch/work/tharvey/catalogs/JAGUAR_SHORT_TEST.fits'
    # instruments = 'Nirc#NIRCam()#+ACS_WFC()+WFC3_IR()# Can leave this - if there is no data for an instrument it is removed automatically
    instruments = [
        "NIRCam"
    ]  # Can leave this - if there is no data for an instrument it is removed automatically
    cat_type = "sex"
    surveys = ["ElGordo"]
    aper_diams = [0.32] * u.arcsec
    zero_point = u.nJy.to(u.ABmag)  # 31.4
    code_names = ["EAZY", "EAZY", "EAZY"]  # , "EAZY"] #[LePhare()]
    eazy_zmax_lowz = [4.0, 6.0]
    eazy_templates = ["fsps_larson", "fsps_jades", "fsps"]
    min_flux_pc_errs = [10, 5]  # , 10]
    excl_bands = []  # ["f606W", "f814W", "f090W", "f115W", "f277W", "f335M", "f356W", "f410M", "f444W"]

    for survey in surveys:
        simulated_pipeline(
            survey,
            fits_cat_path,
            version,
            instruments,
            aper_diams,
            code_names,
            eazy_zmax_lowz,
            min_flux_pc_errs,
            excl_bands,
            cat_type=cat_type,
            eazy_templates=eazy_templates,
            zero_point=zero_point,
        )

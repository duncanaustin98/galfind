#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 00:16:05 2023

@author: austind
"""

# Simulated_Catalogue.py

import numpy as np
from tqdm import tqdm
from . import Simulated_Galaxy
from . import Catalogue_Base
from . import useful_funcs_austind as funcs


class Simulated_Catalogue(Catalogue_Base):
    def __init__(
        self, gals, cat_path, survey_name, cat_creator, sim, codes=[], version=""
    ):
        self.sim = sim
        super().__init__(gals, cat_path, survey_name, cat_creator, codes)

    def from_fits_cat(
        cls,
        fits_cat_path,
        instrument,
        cat_creator,
        code_names,
        low_z_runs,
        survey,
        templates="fsps_larson",
        data=None,
        mask=True,
    ):
        print(__name__)
        # open the catalogue
        fits_cat = funcs.cat_from_path(fits_cat_path)
        # produce galaxy array from each row of the catalogue
        gals = np.array(
            [
                Simulated_Galaxy.from_fits_cat(
                    fits_cat[fits_cat[cat_creator.ID_label] == ID],
                    instrument,
                    cat_creator,
                    [],
                    [],
                )
                for ID in tqdm(
                    np.array(fits_cat[cat_creator.ID_label]),
                    total=len(np.array(fits_cat[cat_creator.ID_label])),
                    desc="Loading galaxies into catalogue",
                )
            ]
        )
        # make catalogue with no SED fitting information
        cat_obj = cls(gals, fits_cat_path, survey, cat_creator)
        if cat_obj != None:
            cat_obj.data = data
        if mask:
            cat_obj.mask(data)
        # run SED fitting for the appropriate code names/low-z runs
        for code_name, low_z_run in zip(code_names, low_z_runs):
            code = getattr(globals()[code_name], code_name)()
            try:  # see whether SED fitting has already been performed
                if low_z_run:
                    low_z_label = "_lowz"
                else:
                    low_z_label = ""
                fits_cat[f"{code.galaxy_property_labels['z']}{low_z_label}"]
            except:
                # perform SED fitting - pretty sure this is broken currently
                cat_obj = code.fit_cat(cat_obj, low_z_run, templates=templates)
        return cat_obj

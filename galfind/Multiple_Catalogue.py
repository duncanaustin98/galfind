
from __future__ import annotations

import astropy.units as u
import json
from itertools import chain
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack, Table
from typing import TYPE_CHECKING, List, Optional, Callable, Union

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

if TYPE_CHECKING:
    from . import (
        Galaxy,
        Catalogue,
        SED_code,
        Selector,
    )

from . import galfind_logger, config
from . import useful_funcs_austind as funcs
from . import Catalogue_Base, Multiple_Filter
from .Catalogue import open_galfind_cat


class Combined_Catalogue_Creator:
    # TODO: Should inherit from 'Catalogue_Creator' parent

    def __init__(
        self: Self,
        survey: str,
        version: str,
        filterset: Multiple_Filter,
        aper_diams: List[u.Quantity],
        cat_path: Optional[str] = None,
        crops: Optional[List[str]] = None,
        open_cat: Optional[Callable[[str, str], Optional[Table]]] = open_galfind_cat,
    ):
        self.survey = survey
        self.version = version
        self.filterset = filterset
        self.aper_diams = aper_diams
        self.cat_path = cat_path
        if crops is None:
            crops = []
        self.crops = crops
        self.open_cat = open_cat

    # Copied and pasted from Catalogue_Creator
    @property
    def crop_name(self) -> List[str]:
        return funcs.get_crop_name(self.crops)


class Combined_Catalogue(Catalogue_Base):
    
    def __init__(
        self: Self,
        gals: List[Galaxy],
        cat_creator: Combined_Catalogue_Creator,
        **kwargs, # for now
    ):
        super().__init__(gals, cat_creator)

        # if not hasattr(self, "cat_arr"):
        #     # TODO: Make cat_arr from galaxy list
        #     # (each galaxy should point to origin catalogue)
        #     pass

    @classmethod
    def from_cats(
        cls,
        cat_arr: List[Catalogue],
        cat_path: Optional[str] = None,
        survey: Optional[str] = None,
        version: Optional[str] = None,
        overwrite: bool = False,
    ):
        # ensure all catalogues have the same aperture diameters
        assert all(cat.aper_diams == cat_arr[0].aper_diams for cat in cat_arr), \
            galfind_logger.critical(
                "All catalogues must have the same aperture diameters"
            )
        
        # determine survey and version if not provided
        if survey is None:
            survey = "+".join(sorted(np.unique([cat.survey for cat in cat_arr])))
        if version is None:
            version = "+".join(sorted(np.unique([cat.version for cat in cat_arr])))
        assert all(isinstance(x, str) for x in [survey, version]), \
            galfind_logger.critical(
                f"Either {survey=} ({type(survey)=}) or {version=} ({type(survey)=}) != str"
            )
        
        # make filterset comprising all available bands
        filters = np.array([filt for filt in chain.from_iterable([[filt for filt in cat.filterset] for cat in cat_arr])])
        unique_filters = filters[sorted(np.unique([filt.band_name for filt in filters], return_index = True)[1])]
        full_cat_filterset = Multiple_Filter(unique_filters)
        full_cat_instr_name = full_cat_filterset.instrument_name
        
        # determine catalogue path
        if cat_path is None:
            assert all(cat.data.forced_phot_band.filt_name == cat_arr[0].data.forced_phot_band.filt_name for cat in cat_arr), \
                galfind_logger.critical(
                    "All catalogues must have the same forced phot band if cat_path is not provided"
                )
            # create the catalogue path by combining names of the other catalogues
            cat_path = funcs.get_phot_cat_path(
                survey,
                version,
                full_cat_instr_name,
                cat_arr[0].aper_diams,
                cat_arr[0].data.forced_phot_band.filt_name,
            )

        if not Path(cat_path).is_file() or overwrite:
            # TODO: Make the loading of these not require to indiviudally specify code names
            from . import SED_code, EAZY, LePhare, Bagpipes
            unique_hdu_names = np.unique(list(chain.from_iterable([cat.get_hdu_names() for cat in cat_arr])))
            full_tab_hdus = np.full(len(unique_hdu_names), None)
            # put flux table first
            assert "OBJECTS" in unique_hdu_names, \
                galfind_logger.critical(
                    "All catalogues must have an 'OBJECTS' HDU"
                )
            unique_hdu_names = np.concatenate((["OBJECTS"], unique_hdu_names[unique_hdu_names != "OBJECTS"]))
            # determine ID column names
            for i, hdu in enumerate(unique_hdu_names):
                # make combined catalogue .fits if this does not already exist
                full_tab_arr = [] #np.full(len(cat_arr), None)
                unique_ids = []
                for j, cat in enumerate(cat_arr):
                    tab = cat.open_cat(hdu = hdu, cropped=False)
                    if i == 0:
                        assert hdu == "OBJECTS", \
                            galfind_logger.critical(
                                "First HDU must be 'OBJECTS'"
                            )
                        tab["SURVEY"] = cat.survey
                        tab["VERSION"] = cat.version
                        tab["INSTR_NAME"] = cat.filterset.instrument_name
                        tab.rename_column(cat.ID_label, "SURVEY_ID")
                    # determine SED_code that the hdu originates from, if not any cat.ID_label
                    ID_colname = [subcls.ID_label for subcls in SED_code.__subclasses__() if subcls.__name__.upper() in hdu]
                    if len(ID_colname) == 0:
                        ID_colname = [cat.ID_label if i != 0 else "SURVEY_ID"]
                    if len(ID_colname) > 1:
                        # choose the first one found
                        ID_colname_hdu_pos = [hdu.find(subcls.__name__.upper()) for subcls in SED_code.__subclasses__() if subcls.__name__.upper() in hdu]
                        ID_colname_index = np.argmin(ID_colname_hdu_pos)
                        ID_colname = [ID_colname[ID_colname_index]]
                    #ID_colname = np.unique(ID_colname)
                    try:
                        assert len(ID_colname) == 1, \
                            galfind_logger.critical(
                                f"Could not determine ID_colname for HDU {hdu}"
                            )
                        ID_colname = ID_colname[0]
                        if j == 0:
                            cat_unique_ids = list(tab[ID_colname])
                        else:
                            cat_unique_ids = np.sum(len(tab_) for tab_ in full_tab_arr) + np.array(list(tab[ID_colname]))
                    except:
                        breakpoint()
                    unique_ids.extend(cat_unique_ids)
                    full_tab_arr.append(tab)
                full_tab = vstack(list(full_tab_arr))
                # TODO: Sort unique IDs!
                full_tab["UNIQUE_ID"] = np.array(unique_ids).astype(np.int32) #np.arange(1, len(full_tab) + 1).astype(np.int32)
                # TODO: sort out metadata - finishing off required
                full_tab.meta["SURVEY"] = survey
                full_tab.meta["VERSION"] = version
                #full_tab.meta["INSTR_NAME"] = full_cat_instr_name
                full_tab_hdus[i] = full_tab
            # save table
            Catalogue_Base.write_cat(full_tab_hdus, unique_hdu_names, cat_path)
            #funcs.make_dirs(cat_path)
            #full_tab.write(cat_path, format="fits")
            #galfind_logger.info(f"Saved combined catalogue to {cat_path}")
        
        # ensure all crops are the same
        assert all(cat.cat_creator.crop_name == cat_arr[0].cat_creator.crop_name for cat in cat_arr), \
            galfind_logger.critical(
                "All catalogues must have the same crops"
            )
        # make combined cat creator
        combined_cat_creator = \
            Combined_Catalogue_Creator(
                survey, 
                version, 
                full_cat_filterset, 
                cat_arr[0].aper_diams,
                cat_path = cat_path,
                crops = cat_arr[0].cat_creator.crops
            )

        gals = list(chain.from_iterable([cat.gals for cat in cat_arr]))
        combined_cat = cls(gals, combined_cat_creator)
        combined_cat.cat_arr = cat_arr
        return combined_cat


    # def save_combined_cat(self, filename):
    #     tables = [cat.open_cat(cropped=True) for cat in self.cat_arr]
    #     for table, cat in zip(tables, self.cat_arr):
    #         table["SURVEY"] = cat.survey

    #     combined_table = vstack(tables)
    #     combined_table.rename_column("NUMBER", "SOURCEX_NUMBER")
    #     combined_table["ID"] = np.arange(1, len(combined_table) + 1)
    #     # Move 'ID' to the first column
    #     try:
    #         new_order = ["ID"] + [
    #             col for col in combined_table.colnames if col != "ID"
    #         ]
    #         combined_table = combined_table[new_order]
    #     except:
    #         pass

    #     combined_table.write(filename, format="fits")
    #     funcs.change_file_permissions(filename)

    # def __add__(self, other):
    #     # Check types to allow adding, Catalogue + Multiple_Catalogue, Multiple_Catalogue + Catalogue, Multiple_Catalogue + Multiple_Catalogue
    #     pass

    # def __and__(self, other):
    #     pass

    # def __len__(self):
    #     return np.sum([len(cat) for cat in self.cat_arr])
    
    # TODO: Make this work appropriately
    # def __getattr__(self, attr):
    #     if attr == "unique_ID":
    #         return self.get_unique_IDs()
    #     else:
    #         super().__getattr__(attr)

    def crop(
        self: Self,
        selector: Type[Selector]
    ) -> Self:
        [cat.crop(selector) for cat in self.cat_arr if selector not in cat.crops]
        self.gals = list(chain.from_iterable([cat.gals for cat in self.cat_arr]))
        return super().crop(selector)

    def _get_unique_IDs(self):
        tab = self.cat_creator.open_cat(self.cat_path, "OBJECTS")
        return [int(tab[np.logical_and((tab["SURVEY_ID"] == gal.ID), \
            (tab["SURVEY"] == gal.survey))]["UNIQUE_ID"]) for gal in self]

    def load_sextractor_ext_src_corrs(self):
        for cat in self.cat_arr:
            cat.load_sextractor_ext_src_corrs()
        # TODO: Check that all galaxies have loaded ext_src_corrs
        # (the galaxies in self.gals should point to those galaxies stored in the catalogue object)
        if not all(hasattr(self.gals[0].aper_phot[aper_diam], "ext_src_corrs") for aper_diam in self.aper_diams):
            raise ValueError("ext_src_corrs not loaded into Galaxy objects")

    def calc_Vmax(
        self: Self,
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        z_step: float = 0.01,
        unmasked_area: Union[str, u.Quantity] = "selection",
    ) -> None:
        # # calculate Vmax for galaxy selection in their origin field
        # [
        #     cat.calc_Vmax(
        #         z_bin, aper_diam, SED_fit_code, z_step
        #     ) for cat in self.cat_arr
        # ]
        # calculate Vmax for galaxy selection in each field
        [
            [
                cat._calc_Vmax(
                    data_cat.data, 
                    z_bin, 
                    aper_diam, 
                    SED_fit_code, 
                    z_step,
                    unmasked_area = unmasked_area,
                ) for data_cat in self.cat_arr
            ] for cat in self.cat_arr
        ]
        # TODO: Check that the above has loaded in appropriate info for the galaxies!
        # # calculate the rest of the Vmax's
        # [
        #     self._calc_Vmax(
        #         cat.data,
        #         z_bin = z_bin,
        #         aper_diam = aper_diam,
        #         SED_fit_code = SED_fit_code,
        #         z_step = z_step,
        #     )
        #     for cat in self.cat_arr
        # ]

        # combine Vmax's for each field on a galaxy by galaxy basis
        save_path = self.get_vmax_ecsv_path(self)
        if not Path(save_path).is_file():
            # make Vmax table
            try:
                Vmax_arr = np.array(
                    [
                        [
                            gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                            V_max[self.crop_name.split("/")[-1]][cat.data.full_name]. \
                            to(u.Mpc**3).value for cat in self.cat_arr
                        ] for gal in self
                    ]
                )
            except:
                breakpoint()
            Vmax_arr = np.where(Vmax_arr == -1.0, 0.0, Vmax_arr)
            Vmax_arr = np.sum(Vmax_arr, axis = 1)
            Vmax_arr = np.where(Vmax_arr == 0.0, -1.0, Vmax_arr)
            data = {
                "ID": np.array([gal.ID for gal in self]),
                "survey": np.array([gal.survey for gal in self]),
                "Vmax": Vmax_arr * u.Mpc ** 3,
            }
            new_tab = Table(data, dtype=[int, str, float])
            new_tab.meta = {"Vmax_invalid_val": -1.0}
            self._save_ecsv(save_path, new_tab)
        breakpoint()
        full_survey_name = funcs.get_full_survey_name(self.survey, self.version, self.filterset)
        tab = Table.read(save_path)
        self._load_Vmax_from_ecsv(tab, aper_diam, SED_fit_code, full_survey_name)
        
    def plot_combined_area_depth(
        self,
        save_path,
        save=False,
        show=False,
        mode="n_nearest",
        aper_diam=0.32 * u.arcsec,
        cmap="viridis",
    ):
        all_array = []
        max_area = 0
        for cat in self.cat_arr:
            cat_creator = cat.cat_creator
            array = cat.data.plot_area_depth(
                cat_creator,
                mode,
                aper_diam,
                show=False,
                save=False,
                return_array=True,
            )
            # array is dict  {band: [area, depth]}
            all_array.append(array)
        # all_array is list of dicts
        # Get all bands
        bands = np.unique(
            [band for array in all_array for band in array.keys()]
        )
        area_band = {band: 0 for band in bands}
        depth_array_band = {band: [] for band in bands}
        for band in bands:
            # Get all areas
            for array in all_array:
                if band in array.keys():
                    area_band[band] += array[band][0]
                    depth_array_band[band].extend(array[band][1])

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_title(f"{self.survey}")
        ax.set_xlabel(r"Area (arcmin$^{2}$)")
        ax.set_ylabel(r"5$\sigma$ Depth (AB mag)")

        colors = cm.get_cmap(cmap)(np.linspace(0, 1, len(bands)))
        for pos, band in enumerate(bands):
            total_depths = np.flip(np.sort(depth_array_band[band]))

            # Calculate the cumulative distribution scaled to area of band
            n = len(total_depths)
            cum_dist = np.arange(1, n + 1) / n
            cum_dist = cum_dist * area_band[band]

            # Plot
            ax.plot(
                cum_dist,
                total_depths,
                label=band if "+" not in band else "Detection",
                color=colors[pos],
                drawstyle="steps-post",
                linestyle="-" if "+" not in band else "--",
            )

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
            print(area_band[band])
            if area_band[band] > max_area:
                max_area = area_band[band]

        ax.set_ylim(max_depth, min_depth - 0.25)
        ax.legend(frameon=False, ncol=2)
        ax.set_xlim(0, max_area)
        ax.grid(True)
        if save:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

    # def __str__(self):
    #     # This should be smarter
    #     return " ".join([str(cat) for cat in self.cat_arr])

    # Need to be able to save fits

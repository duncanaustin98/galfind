# Multiple_Catalogue.py
import astropy.units as u
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack


class Multiple_Catalogue:
    def __init__(self, cat_arr, survey):
        self.cat_arr = cat_arr
        self.survey = survey
        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues
        # self.__rmul__ = self.__mul__

    @classmethod
    def from_pipeline(
        cls,
        survey_list,
        version,
        aper_diams,
        cat_creator,
        code_names,
        lowz_zmax,
        instruments=["NIRCam", "ACS_WFC", "WFC3_IR"],
        forced_phot_band="F444W",
        excl_bands=[],
        loc_depth_min_flux_pc_errs=[5, 10],
        templates_arr=["fsps_larson"],
        select_by=None,
    ):
        cat_arr = [
            Catalogue.from_pipeline(
                survey,
                version,
                aper_diams,
                cat_creator,
                code_names,
                lowz_zmax,
                instruments,
                forced_phot_band,
                excl_bands,
                loc_depth_min_flux_pc_errs,
                templates_arr,
                select_by,
            )
            for survey in survey_list
        ]

        return cls(cat_arr)

    def save_combined_cat(self, filename):
        tables = [cat.open_cat(cropped=True) for cat in self.cat_arr]
        for table, cat in zip(tables, self.cat_arr):
            table["SURVEY"] = cat.survey

        combined_table = vstack(tables)
        combined_table.rename_column("NUMBER", "SOURCEX_NUMBER")
        combined_table["ID"] = np.arange(1, len(combined_table) + 1)
        # Move 'ID' to the first column
        try:
            new_order = ["ID"] + [
                col for col in combined_table.colnames if col != "ID"
            ]
            combined_table = combined_table[new_order]
        except:
            pass

        combined_table.write(filename, format="fits")
        funcs.change_file_permissions(filename)

    def __add__(self, other):
        # Check types to allow adding, Catalogue + Multiple_Catalogue, Multiple_Catalogue + Catalogue, Multiple_Catalogue + Multiple_Catalogue
        pass

    def __and__(self, other):
        pass

    def __len__(self):
        return np.sum([len(cat) for cat in self.cat_arr])

    def calc_UVLF(self):
        pass

    def calc_GSMF(self):
        pass

    def plot(self, x_name, y_name, colour_by, save=False, show=False):
        pass

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

    def __str__(self):
        # This should be smarter
        return " ".join([str(cat) for cat in self.cat_arr])

    # Need to be able to save fits

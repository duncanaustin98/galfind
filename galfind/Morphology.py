
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import os
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import matplotlib as mpl
from astropy.io import fits
import astropy.units as u
from astropy.table import Table
import h5py
from copy import deepcopy
import matplotlib.patheffects as pe
from astropy.visualization import AsinhStretch, ZScaleInterval, ImageNormalize
try:    
    from photutils import EllipticalAperture
except ImportError:
    from photutils.aperture import EllipticalAperture
import subprocess
from typing import Union, Dict, Any, List, Tuple, Callable, Optional, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Galaxy, Catalogue, Catalogue_Base, PSF_Base, Band_Cutout_Base, RGB_Base, Multiple_Cutout_Base, PDF_Base
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger, config
from . import useful_funcs_austind as funcs

fwhm_nircam = {
    "F090W": 33.0 * u.marcsec,
    "F115W": 40.0 * u.marcsec,
    "F150W": 50.0 * u.marcsec,
    "F162M": 55.0 * u.marcsec,
    "F182M": 62.0 * u.marcsec,
    "F200W": 66.0 * u.marcsec,
    "F210M": 71.0 * u.marcsec,
    "F250M": 85.0 * u.marcsec,
    "F277W": 92.0 * u.marcsec,
    "F300M": 100.0 * u.marcsec,
    "F335M": 111.0 * u.marcsec,
    "F356W": 116.0 * u.marcsec,
    "F410M": 137.0 * u.marcsec,
    "F444W": 145.0 * u.marcsec,
}

name_to_label = {
    "n": r"$n$",
    "r_e": r"$r_e$",
    "mag": r"$m_{\mathrm{AB}}$",
    "axr": r"b$/$a",
    "pa": "PA",
    "x_off": r"$x_{\mathrm{off}}$",
    "y_off": r"$y_{\mathrm{off}}$",
    "chi2": r"$\chi^2$",
    "Ndof": r"$N_{\mathrm{dof}}$",
    "red_chi2": r"$\chi^2_{\mathrm{red}}$"
}

class Morphology_Result(ABC):

    def __init__(
        self: Self,
        fitter: Type[Morphology_Fitter],
        chi2: float,
        Ndof: int,
        properties: Dict[str, Union[u.Quantity, u.Magnitude, u.Dex]],
        property_errs: Dict[str, List[Union[u.Quantity, u.Magnitude, u.Dex], Union[u.Quantity, u.Magnitude, u.Dex]]],
        property_pdfs: Dict[str, Type[PDF_Base]],
        rff: Optional[float] = None,
        cutout: Optional[Type[Band_Cutout_Base]] = None,
    ) -> None:
        self.fitter = fitter
        self.chi2 = chi2
        self.Ndof = Ndof
        self.properties = properties
        [setattr(self, key, val) for key, val in properties.items()]
        self.property_errs = property_errs
        [setattr(self, f"{key}_err", val) for key, val in property_errs.items()]
        self.property_pdfs = property_pdfs
        self.rff = rff
        self.cutout = cutout

    @property
    def red_chi2(self: Self) -> float:
        return self.chi2 / self.Ndof
    
    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({self.fitter.name})"
    
    @abstractmethod
    def plot(self: Self) -> None:
        pass


class Galfit_Result(Morphology_Result):

    def __init__(
        self: Self,
        fitter: Type[Galfit_Fitter],
        chi2: float,
        Ndof: int,
        properties: Dict[str, Union[u.Quantity, u.Magnitude, u.Dex]],
        property_errs: Dict[str, List[Union[u.Quantity, u.Magnitude, u.Dex], Union[u.Quantity, u.Magnitude, u.Dex]]],
        property_pdfs: Dict[str, Type[PDF_Base]],
        im_path: str,
        mask_path: Optional[str] = None,
        rff: Optional[float] = None,
        cutout: Optional[Type[Band_Cutout_Base]] = None,
    ) -> None:
        self.im_path = im_path
        self.mask_path = mask_path
        super().__init__(fitter, chi2, Ndof, properties, property_errs, property_pdfs, rff, cutout = cutout)

    @property
    def id(self: Self) -> str:
        return self.im_path.split("/")[-1].split("_")[0]

    @property
    def version(self: Self) -> str:
        return self.im_path.replace(config['GALFIT']['OUTPUT_DIR'], "").split("/")[1]

    @property
    def instr_name(self: Self) -> str:
        return self.im_path.replace(config['GALFIT']['OUTPUT_DIR'], "").split("/")[2]

    @property
    def survey(self: Self) -> str:
        return self.im_path.replace(config['GALFIT']['OUTPUT_DIR'], "").split("/")[3]
    
    @property
    def filt_name(self: Self) -> str:
        return self.im_path.replace(config['GALFIT']['OUTPUT_DIR'], "").split("/")[4]
    
    @property
    def plot_path(self: Self) -> str:
        return f"{'/'.join(self.im_path.replace(config['GALFIT']['OUTPUT_DIR'], config['GALFIT']['GALFIT_PLOT_DIR']).split('/')[:-1]).replace(self.id + '/', '')}/{self.id}.png"

    def plot(
        self: Self,
        fig: Optional[plt.Figure] = None,
        title: Optional[str] = None,
        cmap: str = "gray_r",
        model_cmap: str = "gray_r",
        annotate_properties: List[str] = ["n", "r_e"],
        neighbours: Optional[Dict[str, Any]] = None,
        save: bool = True,
        show: bool = False,
    ) -> None:
        if fig is None:
            fig, axs = plt.subplots(1, 3, figsize=(10, 4))

        hdul = fits.open(self.im_path)
        orig = hdul[1].data # Original image
        mod = hdul[2].data # Galfit model image
        res = hdul[3].data # Residual image
        # Close the fits file
        hdul.close()
        # Open mask image
        mask_hdul = fits.open(self.mask_path)
        mask = mask_hdul[0].data.astype(bool)
        mask_hdul.close()
        seg_data = [hdu for hdu in fits.open(self.cutout.band_data.seg_path) if hdu.name == "SEG"][0].data
        # determine object id as closest seg_data value to x0, y0
        if np.all(seg_data == 0):
            galfind_logger.warning(
                f"{repr(self.cutout)} segmentation map is empty!"
            )
            primary_id = 0
        else:
            nonzero_yx = np.argwhere(seg_data != 0)
            distances = np.hypot(
                nonzero_yx[:, 0] - int((seg_data.shape[0] // 2)),
                nonzero_yx[:, 1] - int((seg_data.shape[1] // 2))
            )
            closest_yx = nonzero_yx[np.argmin(distances)]
            primary_id = seg_data[closest_yx[0], closest_yx[1]]
        unique_ids = np.unique(seg_data)
        segmaps = {}
        for id in unique_ids:
            segmap = deepcopy(seg_data)
            segmap[seg_data == id] = 1
            segmap[seg_data != id] = 0
            segmaps[id] = segmap
        
        # Plot the images
        if title is None:
            # default title
            title = f"ID={self.id}, " + \
                f"{self.filt_name}, {self.survey}, " + \
                f"{self.version}; MODEL={self.fitter.model}; " + \
                r"$\chi^2_{\mathrm{red}}$" + f"={self.red_chi2:.2f}"
        fig.suptitle(title)

        # Set the colour range of the plot to be that of the input image.
        interval = ZScaleInterval(contrast = 0.05)
        #vmin = interval.get_limits(orig[orig > 0])[0]
        #vmax = interval.get_limits(orig)[1]
        vmin, vmax = interval.get_limits(orig)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.1))
        axs[0].imshow(orig, cmap = cmap, norm = norm, origin = "lower")
        axs[1].imshow(mod, cmap = model_cmap, norm = norm, origin = "lower")
        axs[2].imshow(res, cmap = cmap, norm = norm, origin = "lower")
        # overplot primary kron aperture in green on residual
        primary_kron = EllipticalAperture(
            (orig.shape[0] / 2, (orig.shape[1] / 2)),
            self.cutout.meta["A_IMAGE_AS"] / (self.cutout.meta["SIZE_AS"] / self.cutout.meta["SIZE_PIX"]),
            self.cutout.meta["B_IMAGE_AS"] / (self.cutout.meta["SIZE_AS"] / self.cutout.meta["SIZE_PIX"]),
            self.cutout.meta["THETA_IMAGE"]
        )
        for ax in [axs[2]]:
            primary_kron.plot(
                ax=ax,
                edgecolor='darkgreen',
                lw=1.5,
                path_effects = [pe.withStroke(linewidth=2.5, foreground='white')]
            )
            # print RFF value in lower right corner of residual plot
            if self.rff is not None:
                ax.text(
                    0.975, 0.025, f"RFF={self.rff:.3f}",
                    color='darkgreen',
                    fontsize=8,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                    path_effects = [pe.withStroke(linewidth=2.5, foreground='white')]
                )
            
        # overplot crosshairs
        for ax in axs:
            ax.plot(
                orig.shape[0] / 2,
                orig.shape[1] / 2,
                color="purple",
                marker="+",
                ms=8,
                mew=1,
                zorder = 100,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]
            )
            # plot segmap outlines
            if neighbours is not None:
                for id, segmap in segmaps.items():
                    if id == primary_id:
                        ax.contour(segmap.astype(float), colors = "purple", linewidths = 0.5, zorder = 10)
                        ax.contour(segmap.astype(float), colors = "purple", linestyles = "dotted", linewidths = 0.5, zorder = 100)
                    if id in neighbours["ID"]:
                        ax.contour(segmap.astype(float), colors = "dodgerblue", linewidths = 0.3, zorder = 50)

        for ax in [axs[0], axs[2]]:
            # plot masked pixels in red on original and residual
            ax.contour(mask.astype(float), colors='red', linewidths=0.3, zorder = 1)
            orig_hatchcolor = mpl.rcParams['hatch.color']
            mpl.rcParams['hatch.color'] = 'red'
            ax.contourf(mask.astype(float), levels=[0.5, 1.5], hatches=['xxxxxx'], colors='none', linewidths=0.3, zorder = 1)
            mpl.rcParams['hatch.color'] = orig_hatchcolor

        # Set titles and turn off axis
        axs[0].set_title('Original')
        axs[1].set_title('Model')
        axs[2].set_title('Residual')
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            #ax.axis('off')
            # insert black frame around each subplot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0)

        # Annotate the model with fitting parameters
        annotate_kwargs = {
            'color': 'red',
            'fontsize': 8,
            'horizontalalignment': 'left',
            'verticalalignment': 'center',
            'transform': axs[1].transAxes,
            'path_effects': [pe.withStroke(linewidth=0.5, foreground='white')]
        }
        x0, y0, dy = 0.05, 0.95, -0.05
        for name in annotate_properties:
            name = name.lower()
            assert name in self.properties.keys(), \
                galfind_logger.error(f"{name=} not in {self.properties.keys()=}")
            val = self.properties[name].value
            if name in self.property_errs.keys():
                err = self.property_errs[name][0].value
                out_str = rf"{name_to_label[name]}$ = {val:.2f} \pm {err:.2f}$"
            else:
                out_str = rf"{name_to_label[name]}$ = {val:.2f}$"
            axs[1].text(
                x0, 
                y0 + dy * list(self.properties.keys()).index(name), 
                out_str,
                **annotate_kwargs
            )

        # Save the figure
        if save:
            funcs.make_dirs(self.plot_path)
            plt.savefig(self.plot_path, bbox_inches = "tight", dpi=200)
            funcs.change_file_permissions(self.plot_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        galfind_logger.info(
            f"Galfit output image saved to {self.plot_path}"
        )
    

class Morphology_Fitter(ABC):

    def __init__(
        self: Self,
        psf: Type[PSF_Base],
        model: str,
    ) -> None:
        self.psf = psf
        self.model = model.lower()
        assert all(
            model_ in self._available_models
            for model_ in self.model.split("+")
        ), galfind_logger.critical(
            f"Not all {self.model.split('+')} in {self._available_models=}"
        )
    
    @property
    def name(self: Self) -> str:
        return self.__class__.__name__.split("_")[0] + \
            f"_{self.psf.cutout.filt_name}_{self.model}"

    def __call__(
        self: Self,
        object: Union[Galaxy, Type[Catalogue_Base]], #Union[Type[Band_Cutout_Base], Type[RGB_Base], Type[Multiple_Cutout_Base]],
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        from . import Galaxy, Catalogue_Base #Band_Cutout_Base, RGB_Base, Multiple_Cutout_Base
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            result = self._fit_cat(object, *args, **kwargs)
        elif isinstance(object, Galaxy):
            err_message = f"Fitting individual Galaxy objects not yet implemented! Got {repr(object)}"
            galfind_logger.critical(err_message)
            raise NotImplementedError(err_message)
            #result = self._fit_gal(object, *args, **kwargs)
        else:
            raise TypeError(f"{type(object)=} invalid!")
        return result

    @abstractmethod
    def _fit_cat(
        self: Self,
        cat: Catalogue,
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def _fit_cutout(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        cat: Optional[Catalogue] = None,
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        pass

    @property
    @abstractmethod
    def _available_models(self: Self) -> List[str]:
        pass

    def _make_results_table(
        self: Self,
        results: List[Type[Morphology_Result]],
        out_path: str,
    ) -> Table:
        all_property_names = np.unique(np.array([list(result.properties.keys()) for result in results if result is not None]).flatten())
        fit_properties = {name: [result.properties[name].value if result is not None and name in result.properties.keys() else np.nan for result in results] for name in all_property_names}
        fit_property_l1 = {f"{name}_l1": [result.property_errs[name][0].value if result is not None and name in result.property_errs.keys() else np.nan for result in results] for name in all_property_names}
        fit_property_u1 = {f"{name}_u1": [result.property_errs[name][1].value if result is not None and name in result.property_errs.keys() else np.nan for result in results] for name in all_property_names}
        fit_data = {**fit_properties, **fit_property_l1, **fit_property_u1}
        # add ID, chi2, Ndof, red_chi2, and RFF
        for name in ["id", "chi2", "Ndof", "red_chi2", "rff"]:
            if hasattr(results[0], name):
                fit_data[name] = [getattr(result, name) if result is not None else np.nan for result in results]
        tab = Table(fit_data)
        funcs.make_dirs(out_path)
        tab.write(out_path, overwrite=True)
        galfind_logger.info(f"Saved results table to {out_path}")
        return tab


class Galfit_Fitter(Morphology_Fitter):

    model_to_code_dict = {
        "sersic": "ss",
        "sersic+sersic": "ds",
        "sersic+psf": "ss_psf",
        "psf": "psf",
    }
    property_units = {
        "n": u.dimensionless_unscaled,
        "r_e": u.pixel,
        "mag": u.ABmag,
        "axr": u.dimensionless_unscaled,
        "pa": u.deg,
        "x_off": u.pixel,
        "y_off": u.pixel,
    }

    def __init__(
        self: Self,
        psf: Type[PSF_Base],
        model: str,
        primary_constraints: Optional[
            Dict[str, Dict[str, Union[int, float, List[Union[int, float]]]]]
        ] = {
            "sersic": {"x": [-2, 2], "y": [-2, 2], "n": [0.01, 10]},
            "psf": {"x": [-2, 2], "y": [-2, 2]},
        },
        neighbour_constraints: Optional[
            Dict[str, Dict[str, Union[int, float, List[Union[int, float]]]]]
        ] = {
            "sersic": {"x": [-2, 2], "y": [-2, 2], "n": [0.01, 10]},
            "psf": {"x": [-2, 2], "y": [-2, 2]},
        },
        neighbours_model: Optional[str] = "sersic",
        fid_params: Dict[str, Dict[str, Union[int, float]]] = {
            "sersic": {"n": 1, "axr": 1, "pa": 0},
        }
    ) -> None:
        self._load_constraints(
            primary_constraints,
            neighbour_constraints,
            neighbours_model,
        )
        self.fid_params = fid_params
        self._assert_fid_params()
        super().__init__(psf, model)

    def _load_constraints(
        self: Self,
        primary_constraints: Optional[
            Dict[str, Dict[str, Union[int, float, List[Union[int, float]]]]]
        ],
        neighbour_constraints: Optional[
            Dict[str, Dict[str, Union[int, float, List[Union[int, float]]]]]
        ],
        neighbours_model: Optional[str]
    ) -> None:
        self.primary_constraints = primary_constraints
        self.neighbour_constraints = neighbour_constraints
        self.neighbours_model = neighbours_model

        self.fixed_params = {}
        for i, (name, constraints) in enumerate(
            zip(
                ["primary", "neighbour"],
                [self.primary_constraints, self.neighbour_constraints]
            )
        ):
            if i == 1 or self.neighbours_model is None:
                continue
            if constraints is not None:
                for model, params in constraints.items():
                    assert model in self._available_models, \
                        galfind_logger.critical(
                            f"{model=} not in {self._available_models=}"
                        )
                    self.fixed_params[model] = {}
                    for param_key, param_val in params.items():
                        assert param_key in self.property_units.keys() or param_key in ["x", "y"], \
                            galfind_logger.critical(
                                f"{param_key=} not in {self.property_units.keys()=} or ['x', 'y']!"
                            )
                        if isinstance(param_val, list):
                            assert len(param_val) == 2, \
                                galfind_logger.critical(
                                    f"{param_key} constraints must be a list of length 2! Got {param_val=}"
                                )
                            assert param_val[0] < param_val[1], \
                                galfind_logger.critical(
                                    f"{param_key} constraints must be in the form " + \
                                    f"[lower, upper] with lower < upper! Got {param_val=}"
                                )
                        elif isinstance(param_val, (int, float)):
                            self.fixed_params[model][param_key] = param_val
                        else:
                            err_message = f"Invalid constraint value for {param_key} in {model} {name} constraints! " + \
                                f"Must be either a list of length 2 or a single int/float. Got {param_val=}"
                            galfind_logger.critical(err_message)
                            raise TypeError(err_message)

    def _assert_fid_params(self: Self) -> None:
        for model, params in self.fid_params.items():
            assert model in self._available_models, \
                galfind_logger.critical(
                    f"{model=} not in {self._available_models=}"
                )
            if model == "sersic":
                assert all(param in params.keys() for param in ["n", "axr", "pa"]), \
                    galfind_logger.critical(
                        "Fiducial parameters for sersic model must include " + \
                        f"'n', 'axr', and 'pa'! Got {params.keys()=}"
                    )
        # update fiducial parameters with fixed values if applicable
        for model, fixed_params in self.fixed_params.items():
            for param_key, param_val in fixed_params.items():
                if param_key in self.fid_params[model].keys():
                    self.fid_params[model][param_key] = param_val

    @property
    def _available_models(self: Self) -> List[str]:
        return ["sersic", "psf"]

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__call__(object, *args, **kwargs)

    def _fit_cat(
        self: Self,
        cat: Catalogue,
        fit_filt: Filter,
        native: bool = False,
        plot: bool = True,
        *args,
        **kwargs,
    ):
        cat.load_sextractor_params()
        cat.make_band_cutouts(
            fit_filt, 
            self.psf.cutout.cutout_size,
            native = native,
        )
        in_subdir = f"{config['GALFIT']['INPUT_DIR']}/{cat.version}" + \
            f"/{cat.filterset.instrument_name}/" + \
            funcs.aper_diams_to_str(cat.aper_diams).replace('(', '').replace(')', '') + \
            f"/{cat.survey}/{fit_filt.filt_name}"
        out_subdir = f"{config['GALFIT']['OUTPUT_DIR']}/{cat.version}" + \
            f"/{cat.filterset.instrument_name}/" + \
            funcs.aper_diams_to_str(cat.aper_diams).replace('(', '').replace(')', '') + \
            f"/{cat.survey}/{fit_filt.filt_name}"
        cutout_key = f"{fit_filt.filt_name}_" + \
            f"{self.psf.cutout.cutout_size.to(u.arcsec).value:.2f}as"
        if native:
            in_subdir += "_native"
            out_subdir += "_native"
            cutout_key += "_native"
        
        fixed_param_str = ""
        for i, (model, params) in enumerate(self.fixed_params.items()):
            for j, (param_key, param_val) in enumerate(params.items()):
                if j == 0:
                    if fixed_param_str == "":
                        fixed_param_str = "_fixed"
                    fixed_param_str += f"_{model}"
                fixed_param_str += f"_{param_key}={param_val:.1f}"
        
        results = [
            self._fit_cutout(
                gal.cutouts[cutout_key],
                fid_mag = gal.sex_MAG_AUTO[fit_filt.filt_name], 
                fid_re = gal.sex_Re[fit_filt.filt_name],
                in_dir = f"{in_subdir}/{str(gal.ID)}/{self.model}{fixed_param_str}",
                out_dir = f"{out_subdir}/{str(gal.ID)}/{self.model}{fixed_param_str}",
                cat = cat,
                plot = plot,
                *args,
                **kwargs,
            )
            for gal in cat
        ]
        # make output table
        out_path = f"{out_subdir}/{self.model}{fixed_param_str}/results.fits"
        if not Path(out_path).is_file():
            tab = self._make_results_table(results, out_path = out_path)
        else:
            tab = Table.read(out_path)
        # TODO: merge table with existing .fits catalogue pointed to by the Catalogue object
        return tab

    def _fit_cutout(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        fid_mag: u.Magnitude,
        fid_re: u.Quantity,
        in_dir: str = "",
        out_dir: str = "",
        cat: Optional[Catalogue] = None,
        plot: bool = True,
        overwrite: bool = False,
        *args,
        **kwargs,
    ) -> Galfit_Result:
        if in_dir != "":
            in_dir = f"{in_dir}/"
        if out_dir != "":
            out_dir = f"{out_dir}/"
        out_path = self._imgblock_out_path(cutout, out_dir)
        if not Path(out_path).is_file() or overwrite:
            for i, val in enumerate([
                cutout.meta["A_IMAGE_AS"],
                cutout.meta["B_IMAGE_AS"],
                cutout.meta["THETA_IMAGE"],
            ]):
                if (i in [0, 1] and val <= 0.0) or np.isnan(val):
                    galfind_logger.warning(f"{cutout.ID} has invalid sextractor parameters!")
                    return None
            self._make_temps(cutout, out_dir)
            self._make_in_constraints(cutout, out_dir, cat = cat)
            self._make_in_mask(cutout, in_dir, out_dir, cat = cat)
            input_filepath = self._make_input_file(cutout, fid_mag, fid_re, in_dir, cat = cat)
            # TODO: Load results rather than re-running if already performed
            subprocess.run(
                f"{config['GALFIT']['GALFIT_INSTALL_PATH']}/./galfit {input_filepath}", 
                shell=True,
                cwd=out_dir
            )
            self._delete_temps(cutout, out_dir)
            self._move_constraints_to_in_dir(cutout, in_dir, out_dir)
            self._move_mask_to_in_dir(cutout, in_dir, out_dir)
            galfind_logger.info(f"{repr(self)} run for {repr(cutout)}")
        fitting_result, crashed = self._extract_results_from_file(cutout, in_dir, out_dir)

        if not Path(fitting_result.plot_path).is_file() and not crashed and plot:
            if "sersic" in self.model:
                annotate_properties = ["n", "r_e"]
            else:
                annotate_properties = ["mag"]
            fitting_result.plot(
                annotate_properties = annotate_properties,
                neighbours = self._get_neighbours(cutout, in_dir, cat = cat),
                save = True
            )
        # update Cutout object with Morphology results
        cutout.update_morph_fits(fitting_result)
        return fitting_result # TODO: should output cutout

    @staticmethod
    def _temp_img_path(
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{cutout.ID}_img_in.fits"
        funcs.make_dirs(path)
        return path

    @staticmethod
    def _temp_sigma_path(
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{cutout.ID}_sigma_in.fits"
        funcs.make_dirs(path)
        return path

    def _temp_psf_path(
        self: Self,
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{self.psf.cutout.cutout_path.split('/')[-1]}"
        funcs.make_dirs(path)
        return path

    def _make_temps(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> None:
        temp_img_path = Galfit_Fitter._temp_img_path(cutout, out_dir)
        if not Path(temp_img_path).is_file():
            im_data, im_hdr = cutout.band_data.load_im()
            hdul = fits.HDUList([fits.PrimaryHDU(im_data, header=im_hdr)])
            hdul.writeto(temp_img_path, overwrite=True)
        temp_sigma_path = Galfit_Fitter._temp_sigma_path(cutout, out_dir)
        if not Path(temp_sigma_path).is_file():
            rms_err_data, rms_err_hdr = cutout.band_data.load_rms_err(output_hdr = True)
            hdul = fits.HDUList([fits.PrimaryHDU(rms_err_data, header=rms_err_hdr)])
            hdul.writeto(temp_sigma_path, overwrite=True)
        temp_psf_path = self._temp_psf_path(out_dir)
        if not Path(temp_psf_path).is_file():
            os.symlink(self.psf.cutout.cutout_path, temp_psf_path)
    
    @staticmethod
    def _constraints_path(
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{cutout.ID}_constraints_in.txt"
        funcs.make_dirs(path)
        return path

    @staticmethod
    def _mask_path(
        cutout: Type[Band_Cutout_Base],
        out_dir: str = "",
    ) -> str:
        path = f"{out_dir}{cutout.ID}_mask_in.fits"
        funcs.make_dirs(path)
        return path

    @staticmethod
    def _neighbours_path(
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{cutout.ID}_neighbours.h5"
        funcs.make_dirs(path)
        return path

    @staticmethod
    def _imgblock_out_path(
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{cutout.ID}_imgblock_out.fits"
        funcs.make_dirs(path)
        return path

    def _make_in_constraints(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        out_dir: str,
        cat: Optional[Catalogue],
    ) -> None:
        constraints_path = Galfit_Fitter._constraints_path(
            cutout,
            out_dir,
        )
        if not Path(constraints_path).is_file() and self.primary_constraints is not None:
            # make constraints file
            galfind_logger.debug(
                f"Making {repr(self)} constraints file for {repr(cutout)} with " + \
                f"{self.primary_constraints=}, {self.neighbours_model=}, " + \
                f"and {self.neighbour_constraints=}"
            )
            with open(constraints_path, "w") as f:
                # make primary constraints
                model_no = 0
                for model, params in self.primary_constraints.items():
                    if model in self.model:
                        model_no += 1
                        for param_key, param_val in params.items():
                            if isinstance(param_val, list):
                                if model in self.fid_params.keys() and \
                                        param_key in self.fid_params[model].keys():
                                    fid_val = self.fid_params[model][param_key]
                                    param_val = [param_val[0] - fid_val, param_val[1] - fid_val]
                                val_str = f"{param_val[0]} {param_val[1]}"
                                f.write(f"{model_no}       {param_key}       {val_str}\n")

                if self.neighbours_model is not None:
                    neighbours = self._get_neighbours(cutout, out_dir, cat = cat)
                    for i in range(len(neighbours["ID"])):
                        for model, params in self.neighbour_constraints.items():
                            if model in self.neighbours_model:
                                model_no += 1
                                for param_key, param_val in params.items():
                                    if isinstance(param_val, list):
                                        if model in self.fid_params.keys() and \
                                                param_key in self.fid_params[model].keys():
                                            fid_val = self.fid_params[model][param_key]
                                            param_val = [param_val[0] - fid_val, param_val[1] - fid_val]
                                        val_str = f"{param_val[0]} {param_val[1]}"
                                        f.write(f"{model_no}       {param_key}       {val_str}\n")
            galfind_logger.info(f"Saved constraints file to {constraints_path}")
    
    def _make_in_mask(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        in_dir: str = "",
        out_dir: str = "",
        cat: Optional[Catalogue] = None,
    ) -> None:
        mask_path = Galfit_Fitter._mask_path(
            cutout,
            out_dir,
        )
        if not Path(mask_path).is_file():
            # TODO: load properly using cutout.band_data.load_seg()
            seg_data = [hdu for hdu in fits.open(cutout.band_data.seg_path) if hdu.name == "SEG"][0].data
            # determine object id as closest seg_data value to x0, y0
            if np.all(seg_data == 0):
                galfind_logger.warning(
                    f"{repr(cutout)} segmentation map is empty!"
                )
                primary_id = 0
            else:
                nonzero_yx = np.argwhere(seg_data != 0)
                distances = np.hypot(
                    nonzero_yx[:, 0] - int((seg_data.shape[0] // 2)),
                    nonzero_yx[:, 1] - int((seg_data.shape[1] // 2))
                )
                closest_yx = nonzero_yx[np.argmin(distances)]
                primary_id = seg_data[closest_yx[0], closest_yx[1]]
            seg_data[seg_data == primary_id] = 0

            hdr = fits.Header()
            if cat is not None:
                galfind_logger.debug(
                    f"Masking {repr(cutout)} neighbours within Kron aperture"
                )
                neighbours = self._get_neighbours(cutout, in_dir, cat = cat, seg_data = seg_data)
                for neighbour_id in neighbours["ID"]:
                    seg_data[seg_data == neighbour_id] = 0
                hdr["MASK_KRON_NEIGHBOURS"] = False
                for key in neighbours.keys():
                    #if key not in ["KRON"]:
                    hdr[f"NEIGHBOUR_{key}"] = ','.join([str(val) for val in neighbours[key]])
            else:
                galfind_logger.debug(
                    f"No catalogue provided, not masking {repr(cutout)} neighbours"
                )
                hdr["MASK_KRON_NEIGHBOURS"] = True

            seg_data[seg_data > 0] = 1 # mask all detected (non-Kron neighbour) sources

            new_hdu = fits.PrimaryHDU(seg_data, header = hdr)
            funcs.make_dirs(mask_path)
            new_hdu.writeto(mask_path, overwrite=True)
            funcs.change_file_permissions(mask_path)
            galfind_logger.info(f"{repr(cutout)} mask saved as {mask_path}")

    def _get_neighbours(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        in_dir: str = "",
        cat: Optional[Catalogue] = None,
        seg_data: Optional[NDArray[int]] = None,
    ) -> Dict[str, Any]:
        neighbours_path = self._neighbours_path(cutout, in_dir)
        if not Path(neighbours_path).is_file():
            galfind_logger.info(
                f"Extracting {repr(cutout)} neighbours from catalogue for Kron masking"
            )
            if seg_data is None:
                seg_data = [hdu for hdu in fits.open(cutout.band_data.seg_path) if hdu.name == "SEG"][0].data
                # determine object id as closest seg_data value to x0, y0
                if np.all(seg_data == 0):
                    galfind_logger.warning(
                        f"{repr(cutout)} segmentation map is empty!"
                    )
                    primary_id = 0
                else:
                    nonzero_yx = np.argwhere(seg_data != 0)
                    distances = np.hypot(
                        nonzero_yx[:, 0] - int((seg_data.shape[0] // 2)),
                        nonzero_yx[:, 1] - int((seg_data.shape[1] // 2))
                    )
                    closest_yx = nonzero_yx[np.argmin(distances)]
                    primary_id = seg_data[closest_yx[0], closest_yx[1]]
                seg_data[seg_data == primary_id] = 0
            pix_scale = cutout.meta["SIZE_AS"] / cutout.meta["SIZE_PIX"]
            kron_aper = EllipticalAperture(
                (cutout.meta["SIZE_PIX"] / 2, cutout.meta["SIZE_PIX"] / 2),
                cutout.meta["A_IMAGE_AS"] / pix_scale,
                cutout.meta["B_IMAGE_AS"] / pix_scale,
                cutout.meta["THETA_IMAGE"]
            )
            primary_kron_mask = kron_aper.to_mask(method="center")
            primary_source_extent = primary_kron_mask.to_image(seg_data.shape).astype(bool)
            secondary_source_ids = np.unique(seg_data[primary_source_extent])
            secondary_source_ids = secondary_source_ids[secondary_source_ids != 0]
            # include only those sources not touching an image edge
            edge_mask = np.zeros_like(seg_data, dtype = bool)
            edge_mask[0, :] = True
            edge_mask[-1, :] = True
            edge_mask[:, 0] = True
            edge_mask[:, -1] = True
            secondary_source_ids = np.array([
                id for id in secondary_source_ids
                if not np.any(edge_mask[seg_data == id])
            ])
            cutout_band_seg_cat_path = cat.data[cutout.band_data.filt_name].seg_path.replace("_seg.fits", ".fits")
            tab = cat.cat_creator.open_cat(cutout_band_seg_cat_path, "sky_coord")
            wcs = cutout.band_data.load_wcs()
            new_tab = tab[
                np.isin(
                    tab[cat.data[cutout.band_data.filt_name].forced_phot_args["id_label"]],
                    secondary_source_ids
                )
            ]
            secondary_xpos, secondary_ypos = wcs.world_to_pixel(
                cat.cat_creator.load_skycoords_func(
                    new_tab,
                    cat.cat_creator.skycoords_labels,
                    cat.cat_creator.skycoords_units
                )
            )
            # secondary_kron_apers = [
            #     EllipticalAperture(
            #         (xpos, ypos),
            #         row["A_IMAGE"] * row["KRON_RADIUS"],
            #         row["B_IMAGE"] * row["KRON_RADIUS"],
            #         row["THETA_IMAGE"]
            #     ) for row, xpos, ypos in zip(new_tab, secondary_xpos, secondary_ypos)
            # ]
            a_image = new_tab["A_IMAGE"]
            b_image = new_tab["B_IMAGE"]
            kron_radius = new_tab["KRON_RADIUS"]
            theta_image = new_tab["THETA_IMAGE"]
            mag_auto = new_tab["MAG_AUTO"]
            flux_radius = new_tab["FLUX_RADIUS"] * pix_scale
            # write .h5 file for future loading
            with h5py.File(neighbours_path, "w") as f:
                f.create_dataset("ID", data = new_tab[cat.data[cutout.band_data.filt_name].forced_phot_args["id_label"]])
                f.create_dataset("X_IMAGE", data = secondary_xpos)
                f.create_dataset("Y_IMAGE", data = secondary_ypos)
                f.create_dataset("A_IMAGE", data = a_image)
                f.create_dataset("B_IMAGE", data = b_image)
                f.create_dataset("KRON_RADIUS", data = kron_radius)
                f.create_dataset("THETA_IMAGE", data = theta_image)
                f.create_dataset("MAG_AUTO", data = mag_auto)
                f.create_dataset("FLUX_RADIUS", data = flux_radius)
        else:
            # load from .h5
            with h5py.File(neighbours_path, "r") as f:
                secondary_source_ids = f["ID"][:]
                secondary_xpos = f["X_IMAGE"][:]
                secondary_ypos = f["Y_IMAGE"][:]
                a_image = f["A_IMAGE"][:]
                b_image = f["B_IMAGE"][:]
                kron_radius = f["KRON_RADIUS"][:]
                theta_image = f["THETA_IMAGE"][:]
                mag_auto = f["MAG_AUTO"][:]
                flux_radius = f["FLUX_RADIUS"][:]
                # secondary_kron_apers = [
                #     EllipticalAperture(
                #         (xpos, ypos),
                #         a * kron_rad,
                #         b * kron_rad,
                #         theta
                #     ) for xpos, ypos, a, b, kron_rad, theta in zip(
                #         secondary_xpos, 
                #         secondary_ypos, 
                #         a_image, 
                #         b_image, 
                #         kron_radius, 
                #         theta_image
                #     )
                # ]
        return {
            "ID": secondary_source_ids,
            "X_IMAGE": secondary_xpos,
            "Y_IMAGE": secondary_ypos,
            "A_IMAGE": a_image,
            "B_IMAGE": b_image,
            "KRON_RADIUS": kron_radius,
            "THETA_IMAGE": theta_image,
            #"KRON": secondary_kron_apers,
            "MAG_AUTO": mag_auto,
            "FLUX_RADIUS": flux_radius,
        }

    def _make_input_file(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        fid_mag: u.Magnitude,
        fid_re: u.Quantity,
        in_dir: str = "",
        cat: Optional[Catalogue] = None,
    ) -> str:
        size = int((cutout.cutout_size / cutout.band_data.pix_scale).to(u.dimensionless_unscaled).value)
        plate_scale = cutout.band_data.pix_scale.to(u.arcsec).value
        PSF_sampling_factor = 1
        fid_mag_AB = fid_mag.to(u.ABmag).value
        fid_re_pix = (fid_re / cutout.band_data.pix_scale).to(u.dimensionless_unscaled).value
        # make instructions for data
        data_lines = [
            "================================================================================",
            "# IMAGE and GALFIT CONTROL PARAMETERS",
            f"A) {Galfit_Fitter._temp_img_path(cutout, '')}          # input data image (fits file)",
            f"B) {cutout.ID}_imgblock_out.fits   # output data image block",
            f"C) {Galfit_Fitter._temp_sigma_path(cutout, '')}           # Sigma image name",
            f"D) {self.psf.cutout.cutout_path.split('/')[-1]}    # Input PSF image and optional diffusion kernel",
            f"E) {PSF_sampling_factor}   # PSF fine sampling factor relative to data",
            f"F) {Galfit_Fitter._mask_path(cutout, '')}   # Bad pixel mask",
            f"G) {self._constraints_path(cutout, '').split('/')[-1]}   # Constraints file",
            f"H) 0 {size} 0 {size} # Size of cutouts",
            f"I) {size}  {size}   # Size of the convolution box",
            f"J) {cutout.band_data.ZP}   # magnitude photometric zeropoint",
            f"K) {plate_scale} {plate_scale}   # plate scale (dx dy) [arcsec per pixel]",
            f"O) regular   # Display type (regular, curses, both)",
            f"P) 0   # Options: 0=normal run; 1,2=make model/imgblock & quit"
        ]

        # make instructions for primary model
        primary_xy_pos = (size / 2, size / 2)
        primary_model_instr = [
            self._sersic_txt(primary_xy_pos, fid_mag_AB, fid_re_pix) \
            if name == "sersic" else self._psf_txt(primary_xy_pos, fid_mag_AB) \
            for name in self.model.split("+")
        ]
        # make instructions for neighbours if catalogue provided
        if cat is not None:
            # fit neighbours if catalogue provided
            neighbours = self._get_neighbours(cutout, in_dir, cat = cat)
            neighbour_models = []
            for i in range(len(neighbours["ID"])):
                if "sersic" in self.neighbours_model:
                    pix_scale = cutout.meta["SIZE_AS"] / cutout.meta["SIZE_PIX"]
                    neighbour_models.append(
                        self._sersic_txt(
                            (neighbours["X_IMAGE"][i], neighbours["Y_IMAGE"][i]),
                            neighbours["MAG_AUTO"][i],
                            neighbours["FLUX_RADIUS"][i] / pix_scale,
                        )
                    )
                if "psf" in self.neighbours_model:
                    neighbour_models.append(
                        self._psf_txt(
                            (neighbours["X_IMAGE"][i], neighbours["Y_IMAGE"][i]),
                            neighbours["MAG_AUTO"][i]
                        )
                    )
            model_instr = primary_model_instr + neighbour_models
        else:
            model_instr = primary_model_instr
        
        # write these to GALFIT input file
        txt_path = f"{in_dir}{cutout.ID}_{self.model_to_code_dict[self.model]}.txt"
        funcs.make_dirs(txt_path)
        with open(txt_path, "w") as f:
            f.write('\n'.join(data_lines))
            for instr in model_instr:
                f.write('\n' '\n')
                f.write('\n'.join(instr))
            f.close()
        return txt_path

    def _sersic_txt(
        self: Self,
        xy_pos: Tuple[float, float],
        fid_mag_AB: float,
        fid_re_pix: float,
    ) -> List[str]:
        params = ["mag", "r_e", "n", "axr", "pa"]
        if any(param not in params for param in self.fixed_params["sersic"].keys()):
            galfind_logger.critical(f">=1 fixed parameter not in {params=}")
        fixed_params_ = {}
        for name in params:
            if name in self.fixed_params["sersic"].keys():
                galfind_logger.debug(f"Fixing {name} in GALFIT")
                fixed_params_[name] = 0
            else:
                fixed_params_[name] = 1
        return [
            '#Sersic function', 
            f'0)  sersic  # Object type',
            f'1)  {xy_pos[0]} {xy_pos[1]}  1 1   #position x,y [pixel]',
            f'3)  {fid_mag_AB} {fixed_params_["mag"]}   # total mag',
            f'4)  {fid_re_pix}  {fixed_params_["r_e"]}   # effective radius [pixels]',
            f'5)  {self.fid_params["sersic"]["n"]}   {fixed_params_["n"]}   # sersic exponent',
            f'9)  {self.fid_params["sersic"]["axr"]}   {fixed_params_["axr"]} # Axis ratio (b/a) ',
            f'10) {self.fid_params["sersic"]["pa"]}   {fixed_params_["pa"]}',
            f'Z)  0   #  Skip this model in output image?  (yes=1, no=0) '
        ]

    @staticmethod
    def _psf_txt(
        xy_pos: Tuple[float, float],
        fid_mag: float,
    ) -> List[str]:
        return [
            '#PSF function',
            f'0) psf  # Object type',
            f'1) {xy_pos[0]} {xy_pos[1]}  1  1 # position x, y',
            f'3) {fid_mag} 1 # total magnitude',
            f'Z)  0   #  Skip this model in output image?  (yes=1, no=0)'
        ]
    
    def _delete_temps(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> None:
        temp_img_path = Galfit_Fitter._temp_img_path(cutout, out_dir)
        if Path(temp_img_path).is_file():
            os.remove(temp_img_path)
        temp_sigma_path = Galfit_Fitter._temp_sigma_path(cutout, out_dir)
        if Path(temp_sigma_path).is_file():
            os.remove(temp_sigma_path)
        temp_psf_path = self._temp_psf_path(out_dir)
        if Path(temp_psf_path).is_file():
            os.unlink(temp_psf_path)
    
    @staticmethod
    def _move_constraints_to_in_dir(
        cutout: Type[Band_Cutout_Base],
        in_dir: str = "",
        out_dir: str = "",
    ) -> None:
        constraints_path = Galfit_Fitter._constraints_path(
            cutout,
            out_dir,
        )
        if Path(constraints_path).is_file():
            os.rename(
                constraints_path,
                Galfit_Fitter._constraints_path(
                    cutout,
                    in_dir,
                )
            )

    @staticmethod
    def _move_mask_to_in_dir(
        cutout: Type[Band_Cutout_Base],
        in_dir: str = "",
        out_dir: str = "",
    ) -> None:
        mask_path = Galfit_Fitter._mask_path(
            cutout,
            out_dir,
        )
        if Path(mask_path).is_file():
            os.rename(
                mask_path,
                Galfit_Fitter._mask_path(
                    cutout,
                    in_dir,
                )
            )
    
    def _extract_results_from_file(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        in_dir: str,
        out_dir: str,
        pdf_len: int = 10_000,
    ) -> Tuple[Galfit_Result, bool]:
        
        out_path = self._imgblock_out_path(cutout, out_dir)
        property_names = {}
        out_properties = {}
        out_property_errs = {}
        chi2 = {}
        Ndof = {}
        
        # model_names = self.model.split("+")
        # model_results = [
        #     self._extract_psf_results(cutout, out_path) if name == "psf"
        #     else self._extract_sersic_results(cutout, out_path)
        #     for name in model_names
        # ]
        if "psf" in self.model:
            property_names, out_properties, out_property_errs, chi2, Ndof = self._extract_psf_results(cutout, out_path)
        #breakpoint()
        if "sersic" in self.model:
            property_names, out_properties, out_property_errs, chi2, Ndof = self._extract_sersic_results(cutout, out_path)
        #breakpoint()
        
        out_properties = np.array(out_properties, dtype=float)
        out_property_errs = np.array(out_property_errs, dtype=float)

        if all(np.isnan(val) for val in out_properties):
            crashed = True
        else:
            crashed = False
        
        # Make (and output) fitting result object
        from . import PDF
        # unordered PDFs - i.e. chain 1 for sersic is not the same as chain 1 for r_e
        properties = {
            name: float(val) * Galfit_Fitter.property_units[name] \
            for name, val in zip(property_names, out_properties)
        }
        property_errs = {
            name: [
                float(val) * Galfit_Fitter.property_units[name], \
                float(val) * Galfit_Fitter.property_units[name]
            ] for name, val in zip(property_names, out_property_errs)
        }
        try:
            pdfs = {
                name: PDF.from_1D_arr(name, np.random.normal(properties[name].value, \
                property_errs[name][0].value, pdf_len) * properties[name].unit) \
                if np.isfinite(property_errs[name][0].value) else None \
                #np.full(pdf_len, properties[name].value) * properties[name].unit) \
                for name in property_names
            }
        except Exception as e:
            galfind_logger.error(f"Error creating PDFs: {e}")
            breakpoint()
            raise Exception(e)
        if np.isnan(chi2):
            rff = np.nan
        else:
            rff = self._calc_RFF(cutout, in_dir, out_dir)

        return Galfit_Result(
            fitter = self,
            chi2 = chi2,
            Ndof = Ndof,
            properties = properties,
            property_errs = property_errs,
            property_pdfs = pdfs,
            im_path = out_path,
            mask_path = Galfit_Fitter._mask_path(cutout, in_dir),
            rff = rff,
            cutout = cutout,
        ), crashed

        
    def _extract_psf_results(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        out_path: str,
    ) -> Galfit_Result:
        property_names = ["mag", "x_off", "y_off"]
        if Path(out_path).is_file():
            hdr = fits.open(out_path)[2].header
            cen_x, cen_x_err = hdr['1_XC'].split('+/-')
            cen_x = cen_x.replace('*', '')
            cen_x_err = cen_x_err.replace('*', '')
            cen_y, cen_y_err = hdr['1_YC'].split('+/-')
            cen_y = cen_y.replace('*', '')
            cen_y_err = cen_y_err.replace('*', '')
            # extract sersic index
            mag = hdr['1_MAG']
            # if galfit was ran with fixed n=1, then the n value will be in brackets
            if '[' in str(mag):
                mag = float(mag.replace("[", "").replace("]", ""))
                mag_err = float(0.0)
            else:
                mag, mag_err = mag.split('+/-')
                mag = mag.replace('*', '')
                mag_err = mag_err.replace('*', '')
            size = int((cutout.cutout_size / cutout.band_data.pix_scale).to(u.dimensionless_unscaled).value)
            x_off = float(cen_x) - size / 2
            y_off = float(cen_y) - size / 2
            out_properties = [mag, x_off, y_off]
            out_property_errs = [mag_err, cen_x_err, cen_y_err]
            chi_sq = float(hdr["CHISQ"])
            Ndof = int(hdr["NDOF"])
        else:
            out_properties = np.full(len(property_names), np.nan)
            out_property_errs = np.full(len(property_names), np.nan)
            chi_sq = np.nan
            Ndof = np.nan
        assert len(out_properties) == len(property_names), \
            galfind_logger.critical(
                f"{len(out_properties)=}!={len(property_names)=}; " + \
                f"{out_properties=}, {property_names=}"
            )
        assert len(out_property_errs) == len(property_names), \
            galfind_logger.critical(
                f"{len(out_property_errs)=}!={len(property_names)=}; " + \
                f"{out_property_errs=}, {property_names=}"
            )
        return property_names, out_properties, out_property_errs, chi_sq, Ndof

    def _extract_sersic_results(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        out_path: str,
    ) -> Galfit_Result:
        property_names = list(Galfit_Fitter.property_units.keys()) #["n", "r_e", "mag", "axr", "pa", "x_off", "y_off"]
        if Path(out_path).is_file():
            hdr = fits.open(out_path)[2].header
            mag, mag_err = hdr['1_MAG'].split('+/-')
            mag = mag.replace('*', '')
            mag_err = mag_err.replace('*', '')
            re, re_err = hdr['1_RE'].split('+/-')
            re = re.replace('*', '')
            re_err = re_err.replace('*', '')
            axr, axr_err = hdr['1_AR'].split('+/-')
            axr = axr.replace('*', '')
            axr_err = axr_err.replace('*', '')  
            pa, pa_err = hdr['1_PA'].split('+/-')
            pa = pa.replace('*', '')
            pa_err = pa_err.replace('*', '')
            cen_x, cen_x_err = hdr['1_XC'].split('+/-')
            cen_x = cen_x.replace('*', '')
            cen_x_err = cen_x_err.replace('*', '')
            cen_y, cen_y_err = hdr['1_YC'].split('+/-')
            cen_y = cen_y.replace('*', '')
            cen_y_err = cen_y_err.replace('*', '')
            size = int((cutout.cutout_size / cutout.band_data.pix_scale).to(u.dimensionless_unscaled).value)
            x_off = float(cen_x) - size / 2
            y_off = float(cen_y) - size / 2
            # extract sersic index
            n = hdr['1_N']
            # if galfit was ran with fixed n=1, then the n value will be in brackets
            if '[' in str(n):
                n = float(n.replace("[", "").replace("]", ""))
                n_err = float(0.0)
            else:
                n, n_err = n.split('+/-')
                n = n.replace('*', '')
                n_err = n_err.replace('*', '')
            out_properties = [n, re, mag, axr, pa, x_off, y_off]
            out_property_errs = [n_err, re_err, mag_err, axr_err, pa_err, cen_x_err, cen_y_err]
            chi_sq = float(hdr["CHISQ"])
            Ndof = int(hdr["NDOF"])
        else:
            out_properties = np.full(len(property_names), np.nan)
            out_property_errs = np.full(len(property_names), np.nan)
            chi_sq = np.nan
            Ndof = np.nan
        assert len(out_properties) == len(property_names), \
            galfind_logger.critical(
                f"{len(out_properties)=}!={len(property_names)=}; " + \
                f"{out_properties=}, {property_names=}"
            )
        assert len(out_property_errs) == len(property_names), \
            galfind_logger.critical(
                f"{len(out_property_errs)=}!={len(property_names)=}; " + \
                f"{out_property_errs=}, {property_names=}"
            )
        return property_names, out_properties, out_property_errs, chi_sq, Ndof
    
    def _calc_RFF(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        in_dir: str,
        out_dir: str,
    ) -> float:
        """
        Calculates the Residual Flux Fraction (RFF) for a given galaxy.
        """
        # TODO: Cannot re-create the FLUX_AUTO values from the Kron aperture 
        # -> Kron radius too small
        # -> elliptical aperture also too small
        fit_hdul = fits.open(self._imgblock_out_path(cutout, out_dir))
        data = fit_hdul[1].data
        model = fit_hdul[2].data
        #err = cutout.band_data.load_rms_err()

        pix_scale = cutout.meta["SIZE_AS"] / cutout.meta["SIZE_PIX"]
        # reconstruct Kron elliptical aperture
        kron_aper = EllipticalAperture(
            (cutout.meta["SIZE_PIX"] / 2, cutout.meta["SIZE_PIX"] / 2),
            cutout.meta["A_IMAGE_AS"] / pix_scale,
            cutout.meta["B_IMAGE_AS"] / pix_scale,
            cutout.meta["THETA_IMAGE"],
        )
        residuals = abs(data - model)
        model_kron = kron_aper.do_photometry(model)[0][0]
        residual_kron = kron_aper.do_photometry(residuals)[0][0]

        # open cutout segmentation map and mask all sources except those within Kron aperture
        mask = [hdu for hdu in fits.open(cutout.band_data.seg_path) if hdu.name == "SEG"][0].data
        mask[mask != 0] = 1
        bkg_values = data[~mask.astype(bool)]
        abs_deviation = np.abs(bkg_values - np.nanmedian(bkg_values))
        depth_1sig = 1.4826 * np.nanmedian(abs_deviation)
        
        # counter = 0
        # bkg = []
        # while counter < 10:
        #     # Define the size of the apertures
        #     aperture_size = 10

        #     #Generate random coordinates within the image's dimensions
        #     x = np.random.randint(0, mask.shape[1] - 1)
        #     y = np.random.randint(0, mask.shape[0] - 1)
        
        #     if x - (aperture_size/2) >= 0 and x + (aperture_size/2) < mask.shape[1] and y - (aperture_size/2) >= 0 and y + (aperture_size/2) < mask.shape[0]:
        #         #print(x + 10, x - 10)
        #         # Calculate the coordinates of the aperture
        #         x_start = max(0, x - aperture_size // 2)
        #         y_start = max(0, y - aperture_size // 2)
        #         x_end = min(mask.shape[1], x + aperture_size // 2)
        #         y_end = min(mask.shape[0], y + aperture_size // 2)
                
        #         # Extract the pixels within the aperture
        #         aperture = mask[y_start:y_end, x_start:x_end]
                
        #         # Check if all pixels within the aperture are 0
        #         if np.all(aperture == 0):
        #             aperture = err[x_start:x_end, y_start:y_end] 
        #             mean_aperture = sum(sum(aperture)) / (len(aperture)**2)
        #             bkg.append(mean_aperture)
        #             counter += 1
        # bkg_mean = np.mean(bkg) #data[~mask.astype(bool)].mean()

        rff = (residual_kron - 0.8 * depth_1sig * kron_aper.area) / model_kron
        return rff
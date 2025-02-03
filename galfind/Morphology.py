
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.table import Table
import matplotlib.patheffects as pe
from photutils import EllipticalAperture
import subprocess
from typing import Union, Dict, Any, List, Tuple, Callable, Optional, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Galaxy, Catalogue, PSF_Base, Band_Cutout_Base, RGB_Base, Multiple_Cutout_Base, PDF_Base
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger, config
from . import useful_funcs_austind as funcs

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
        rff: Optional[float] = None,
    ) -> None:
        self.im_path = im_path
        #f'{galaxy_path}/{id}_ss_imgblock.fits'
        super().__init__(fitter, chi2, Ndof, properties, property_errs, property_pdfs, rff)

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
        cmap: str = "gray",
        annotate_properties: List[str] = ["n", "r_e"],
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

        # Set the colour range of the plot to be that of the input image.
        vmin = orig.min()
        vmax = orig.max()
        
        # Plot the images
        if title is None:
            # default title
            title = f"ID={self.id}, " + \
                f"{self.filt_name}, {self.survey}, " + \
                f"{self.version}; MODEL={self.fitter.model}; " + \
                r"$\chi^2_{\mathrm{red}}$" + f"={self.red_chi2:.2f}"
        fig.suptitle(title)
        axs[0].imshow(orig, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].imshow(mod, cmap=cmap)
        axs[2].imshow(res, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Set titles and turn off axis
        axs[0].set_title('Original')
        axs[1].set_title('Model')
        axs[2].set_title('Residual')
        for ax in axs.flat:
            ax.axis('off')

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
        assert self.model in self._available_models, \
            galfind_logger.critical(
                f"{self.model=} not in {self._available_models=}"
            )
    
    @property
    def name(self: Self) -> str:
        return self.__class__.__name__.split("_")[0] + \
            f"_{self.psf.cutout.filt_name}_{self.model}"

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue], #Union[Type[Band_Cutout_Base], Type[RGB_Base], Type[Multiple_Cutout_Base]],
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        from . import Galaxy, Catalogue #Band_Cutout_Base, RGB_Base, Multiple_Cutout_Base
        if isinstance(object, Catalogue):
            result = self._fit_cat(object, *args, **kwargs)
        elif isinstance(object, Galaxy):
            result = self._fit_gal(object, *args, **kwargs)
        else:
            raise TypeError(f"{type(object)=} invalid!")
        return result
        # if isinstance(object, tuple(Band_Cutout_Base.__subclasses__())):
        #     self._fit_gal_single_band(object)
        # elif isinstance(object, tuple(RGB_Base.__subclasses__())):
        #     self._fit_gal_rgb(object)
        # elif isinstance(object, tuple(Multiple_Cutout_Base.__subclasses__())):
        #     self._fit_cat(object)

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
        all_property_names = np.unique(np.array([list(result.properties.keys()) for result in results]).flatten())
        fit_properties = {name: [result.properties[name].value if name in result.properties.keys() else np.nan for result in results] for name in all_property_names}
        fit_property_l1 = {f"{name}_l1": [result.property_errs[name][0].value if name in result.property_errs.keys() else np.nan for result in results] for name in all_property_names}
        fit_property_u1 = {f"{name}_u1": [result.property_errs[name][1].value if name in result.property_errs.keys() else np.nan for result in results] for name in all_property_names}
        fit_data = {**fit_properties, **fit_property_l1, **fit_property_u1}
        # add ID, chi2, Ndof, red_chi2, and RFF
        for name in ["id", "chi2", "Ndof", "red_chi2", "rff"]:
            if hasattr(results[0], name):
                fit_data[name] = [getattr(result, name) for result in results]
        tab = Table(fit_data)
        funcs.make_dirs(out_path)
        tab.write(out_path, overwrite=True)
        galfind_logger.info(f"Saved results table to {out_path}")
        return tab


class Galfit_Fitter(Morphology_Fitter):

    model_to_code_dict = {"sersic": "ss", "sersic+sersic": "ds", "sersic+psf": "ss_psf"}
    property_units = {
        "n": u.dimensionless_unscaled,
        "r_e": u.pixel,
        "mag": u.ABmag,
        "axr": u.dimensionless_unscaled,
        "pa": u.deg,
        "x_off": u.pixel,
        "y_off": u.pixel
    }

    def __init__(
        self: Self,
        psf: Type[PSF_Base],
        model: str,
    ) -> None:
        super().__init__(psf, model)

    @property
    def _available_models(self: Self) -> List[str]:
        return ["sersic", "sersic+sersic", "sersic+psf"]

    @property
    def constraints_path(self):
        path = f"{config['GALFIT']['CONSTRAINTS_DIR']}/constraints.txt"
        funcs.make_dirs(path)
        return path

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue], #Union[Type[Band_Cutout_Base], Type[RGB_Base], Type[Multiple_Cutout_Base]],
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        self._make_constraints_file()
        super().__call__(object, *args, **kwargs)

    def _make_constraints_file(self):
        # Create constraints file for Galfit
        if not Path(self.constraints_path).is_file():
            with open(self.constraints_path, "w") as f:
                f.write('\n'.join(["1       x       -10 10", "1     y      -10 10"]))
            galfind_logger.info(f"Saved constraints file to {self.constraints_path}")

    def _fit_cat(
        self: Self,
        cat: Catalogue,
        plot: bool = True,
    ):
        cat.load_sextractor_auto_mags()
        cat.load_sextractor_Re()
        from . import Catalogue_Cutouts
        cat.make_band_cutouts(
            self.psf.cutout.band_data.filt, 
            self.psf.cutout.cutout_size
        )
        in_subdir = f"{config['GALFIT']['INPUT_DIR']}/{cat.version}" + \
            f"/{cat.filterset.instrument_name}/" + \
            f"{cat.survey}/{self.psf.cutout.band_data.filt_name}"
        out_subdir = f"{config['GALFIT']['OUTPUT_DIR']}/{cat.version}" + \
            f"/{cat.filterset.instrument_name}/" + \
            f"{cat.survey}/{self.psf.cutout.band_data.filt_name}"
        cutout_key = f"{self.psf.cutout.band_data.filt_name}_" + \
            f"{self.psf.cutout.cutout_size.to(u.arcsec).value:.2f}as"
        results = [
            self._fit_cutout(
                gal.cutouts[cutout_key],
                fid_mag = gal.sex_MAG_AUTO[self.psf.cutout.band_data.filt_name], 
                fid_re = gal.sex_Re[self.psf.cutout.band_data.filt_name],
                in_dir = f"{in_subdir}/{str(gal.ID)}",
                out_dir = f"{out_subdir}/{str(gal.ID)}/{self.model}",
            )
            for gal in cat
        ]
        # make output table
        out_path = f"{out_subdir}/{self.model}/results.fits"
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
    ) -> Galfit_Result:
        if in_dir != "":
            in_dir = f"{in_dir}/"
        if out_dir != "":
            out_dir = f"{out_dir}/"
        out_path = self._imgblock_out_path(cutout, out_dir)
        if not Path(out_path).is_file():
            self._make_temps(cutout, out_dir)
            self._make_in_mask(cutout, out_dir)
            input_filepath = self._make_input_file(cutout, fid_mag, fid_re, in_dir)
            # TODO: Load results rather than re-running if already performed
            subprocess.run(
                f"{config['GALFIT']['GALFIT_INSTALL_PATH']}/./galfit {input_filepath}", 
                shell=True, 
                cwd=out_dir
            )
            self._delete_temps(cutout, out_dir)
            self._move_mask_to_in_dir(cutout, in_dir, out_dir)
            galfind_logger.info(f"{cutout.ID} Galfit run finished")
        fitting_result = self._extract_results_from_file(cutout, in_dir, out_dir)

        if not Path(fitting_result.plot_path).is_file():
            fitting_result.plot()
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
    
    def _temp_constraints_path(
        self: Self,
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{self.constraints_path.split('/')[-1]}"
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
        temp_constraints_path = self._temp_constraints_path(out_dir)
        if not Path(temp_constraints_path).is_file():
            os.symlink(self.constraints_path, temp_constraints_path)
    
    @staticmethod
    def _mask_path(
        cutout: Type[Band_Cutout_Base],
        out_dir: str = ""
    ) -> str:
        path = f"{out_dir}{cutout.ID}_mask_in.fits"
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
    
    def _make_in_mask(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        out_dir: str = "",
    ) -> None:
        mask_path = Galfit_Fitter._mask_path(cutout, out_dir)
        if not Path(mask_path).is_file():
            # TODO: load properly using cutout.band_data.load_seg()
            seg_data = [hdu for hdu in fits.open(cutout.band_data.seg_path) if hdu.name == "SEG"][0].data
            x0 = int((seg_data.shape[0] / 2) - 1)
            y0 = int((seg_data.shape[1] / 2) - 1)
            seg_data[seg_data == seg_data[x0, y0]] = 0 # change object to 0
            seg_data[seg_data > 0] = 1 # change other objects to 1
            new_hdu = fits.PrimaryHDU(seg_data)
            new_hdu.writeto(mask_path, overwrite=True)
            #galfind_logger.info(f"{cutout.ID} mask saved as {mask_path}")

    def _make_input_file(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        fid_mag: u.Magnitude,
        fid_re: u.Quantity,
        in_dir: str = "",
    ) -> str:
        size = int((cutout.cutout_size / cutout.band_data.pix_scale).to(u.dimensionless_unscaled).value)
        plate_scale = cutout.band_data.pix_scale.to(u.arcsec).value
        PSF_sampling_factor = 1
        sersic_index = 1
        axis_ratio = 1
        PA = 0
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
            f"G) {self.constraints_path.split('/')[-1]}   # Constraints file",
            f"H) 0 {size} 0 {size} # Size of cutouts",
            f"I) {size}  {size}   # Size of the convolution box",
            f"J) {cutout.band_data.ZP}   # magnitude photometric zeropoint",
            f"K) {plate_scale} {plate_scale}   # plate scale (dx dy) [arcsec per pixel]",
            f"O) regular   # Display type (regular, curses, both)",
            f"P) 0   # Options: 0=normal run; 1,2=make model/imgblock & quit"
        ]
        # make instructions for model
        model_lines_arr = [self._sersic_txt(size, fid_mag_AB, fid_re_pix, sersic_index, axis_ratio, PA) \
            if name == "sersic" else self._psf_txt(size, fid_mag) \
            for name in self.model.split("+")]

        # write these to GALFIT input file
        txt_path = f"{in_dir}{cutout.ID}_{self.model_to_code_dict[self.model]}.txt"
        funcs.make_dirs(txt_path)
        with open(txt_path, "w") as f:
            f.write('\n'.join(data_lines))
            for model_lines in model_lines_arr:
                f.write('\n' '\n')
                f.write('\n'.join(model_lines))
            f.close()
        return txt_path

    @staticmethod
    def _sersic_txt(
        size: int, 
        fid_mag_AB: float,
        fid_re_pix: float,
        sersic_index: Union[int, float],
        axis_ratio: Union[int, float],
        PA: Union[int, float],
    ) -> List[str]:
        return [ '#Sersic function', 
            f'0)  sersic  # Object type',
            f'1)  {size//2} {size//2}  1 1   #position x,y [pixel]',
            f'3)  {fid_mag_AB} 1   # total mag',
            f'4)  {fid_re_pix}  1   # effective radius [pixels]',
            f'5)  {sersic_index}   1   # sersic exponent',
            f'9)  {axis_ratio}   1 # Axis ratio (b/a) ',
            f'10) {PA}   1',
            f'Z)  0   #  Skip this model in output image?  (yes=1, no=0) '
        ]

    @staticmethod
    def _psf_txt(
        size: int,
        fid_mag: float,
    ) -> List[str]:
        return [
            '#PSF function',
            f'0) psf  # Object type',
            f'1) {size//2} {size//2}  1  1 # position x, y',
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
        temp_constraints_path = self._temp_constraints_path(out_dir)
        if Path(temp_constraints_path).is_file():
            os.unlink(temp_constraints_path)
    
    @staticmethod
    def _move_mask_to_in_dir(
        cutout: Type[Band_Cutout_Base],
        in_dir: str = "",
        out_dir: str = ""
    ) -> None:
        mask_path = Galfit_Fitter._mask_path(cutout, out_dir)
        if Path(mask_path).is_file():
            os.rename(mask_path, Galfit_Fitter._mask_path(cutout, in_dir))
    
    def _extract_results_from_file(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        in_dir: str,
        out_dir: str,
        pdf_len: int = 10_000,
    ) -> Galfit_Result:
        out_path = self._imgblock_out_path(cutout, out_dir)
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
        # TODO: RFF calculation

        # Make (and output) fitting result object
        from . import PDF
        # unordered PDFs - i.e. chain 1 for sersic is not the same as chain 1 for r_e
        property_names = list(Galfit_Fitter.property_units.keys()) #["n", "r_e", "mag", "axr", "pa", "x_off", "y_off"]
        properties = {name: float(val) * Galfit_Fitter.property_units[name] \
            for name, val in zip(property_names, [n, re, mag, axr, pa, x_off, y_off])}
        property_errs = {name: [float(val) * Galfit_Fitter.property_units[name], \
            float(val) * Galfit_Fitter.property_units[name]] for name, val in \
            zip(property_names, [n_err, re_err, mag_err, axr_err, pa_err, cen_x_err, cen_y_err])}
        pdfs = {name: PDF.from_1D_arr(name, np.random.normal(properties[name].value, \
            property_errs[name][0].value, pdf_len) * properties[name].unit \
            if not np.isnan(property_errs[name][0].value) else \
            np.full(pdf_len, properties[name].value) * properties[name].unit) \
            for name in property_names}
        rff = self._calc_RFF(cutout, in_dir, out_dir)
        return Galfit_Result(
            fitter = self,
            chi2 = float(hdr["CHISQ"]),
            Ndof = int(hdr["NDOF"]),
            properties = properties,
            property_errs = property_errs,
            property_pdfs = pdfs,
            im_path = out_path,
            rff = rff
        )
    
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
        err = cutout.band_data.load_rms_err()
        mask = fits.open(Galfit_Fitter._mask_path(cutout, in_dir))[0].data
        pix_scale = cutout.meta["SIZE_AS"] / cutout.meta["SIZE_PIX"]
        
        # reconstruct Kron elliptical aperture
        kron_aper = EllipticalAperture(
            ((cutout.meta["SIZE_PIX"] - 1) / 2, (cutout.meta["SIZE_PIX"] - 1) / 2),
            cutout.meta["A_IMAGE_AS"] / pix_scale,
            cutout.meta["B_IMAGE_AS"] / pix_scale,
            cutout.meta["THETA_IMAGE"]
        )
        residuals = abs(data - model)
        model_kron = kron_aper.do_photometry(model)[0][0]
        residual_kron = kron_aper.do_photometry(residuals)[0][0]
        # Depths.make_grid(data, mask, 
        # background_aper = 
    
        counter = 0
        background = []
        while counter < 10:
            # Define the size of the apertures
            aperture_size = 10

            #Generate random coordinates within the image's dimensions
            x = np.random.randint(0, mask.shape[1] - 1)
            y = np.random.randint(0, mask.shape[0] - 1)
        
            if x - (aperture_size/2) >= 0 and x + (aperture_size/2) < mask.shape[1] and y - (aperture_size/2) >= 0 and y + (aperture_size/2) < mask.shape[0]:
                #print(x + 10, x - 10)
                # Calculate the coordinates of the aperture
                x_start = max(0, x - aperture_size // 2)
                y_start = max(0, y - aperture_size // 2)
                x_end = min(mask.shape[1], x + aperture_size // 2)
                y_end = min(mask.shape[0], y + aperture_size // 2)
                
                # Extract the pixels within the aperture
                aperture = mask[y_start:y_end, x_start:x_end]
                
                # Check if all pixels within the aperture are 0
                if np.all(aperture == 0):
                    aperture = err[x_start:x_end, y_start:y_end]
                    mean_aperture = sum(sum(aperture)) / (len(aperture)**2)
                    background.append(mean_aperture)
                    counter = counter + 1
            
        background_mean = sum(background) / len(background)
        rff = (residual_kron - 0.8 * background_mean * kron_aper.area) / model_kron
        return rff

# def input_images(galaxy_path, save_path, id, field, band):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     galaxy = fits.open(f'{galaxy_path}/{id}.fits')
#     err = fits.open(f'{galaxy_path}/{id}sigma.fits')
#     mask = fits.open(f'{galaxy_path}/{id}mask_final.fits')
#     psf = fits.open(f'{galaxy_path}/PSF_Resample_{band}_scaled.fits')


#     fig, axs = plt.subplots(2,2, figsize=(10, 4))
#     axs[0,0].imshow(galaxy[0].data, cmap='gray')
#     axs[0,1].imshow(err[0].data, cmap='gray')
#     axs[1,0].imshow(mask[0].data, cmap='gray')
#     axs[1,1].imshow(psf[0].data, cmap='gray')

#     axs[0,0].set_title('Science Image')
#     axs[0,1].set_title('Weight Map')
#     axs[1,0].set_title('Bad Pixel Mask')
#     axs[1,1].set_title('PSF')

#     for ax in axs.flat:
#         ax.axis('off')  # Turn off the axis labels

#     fig.suptitle(f'{field} {id} {band}')  # Add a title to the figure

#     plt.savefig(f'{save_path}/{id}_{field}_input.png', dpi=200)
#     #print(f'{id} input images saved')
#     plt.close(fig)

# deblends nearby sources!!!
# def mask_creation(ring, band, output_path):
#     science_image = imports.fits.open(f'{output_path}/{ring}_{band.lower()}_homog2.fits')[0].data
#     error_image = imports.fits.open(f'{output_path}/{ring}_{band.lower()}_err.fits')[0].data
#     threshold = 2 * error_image
#     segment_map = imports.detect_sources(science_image, threshold, npixels=10)
#     segmentation_deblend = imports.deblend_sources(science_image, segment_map, npixels=10, nlevels=32, contrast=0.125, progress_bar=True)  # Increase contrast value
#     imports.fits.writeto(f'{output_path}/{ring}_{band.lower()}_segmap.fits', segmentation_deblend.data, overwrite=True)

#     hdul = imports.fits.open(f'{output_path}/{ring}_{band.lower()}_segmap.fits')
#     data = hdul[0].data
#     x0 = int((data.shape[0]/2) - 1)
#     y0 = x0
#     data[data == data[x0,y0]] = 0  #change object to 0
#     data[data > 0 ] = 1 #change other objects to 1
#     imports.fits.writeto(f'{output_path}/{ring}_{band.lower()}_mask_final.fits', data, overwrite=True)
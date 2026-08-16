from __future__ import annotations

import os
import shutil
import glob
import numpy as np
from numpy.typing import NDArray
import traceback
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
import subprocess
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging
from typing import Optional, Dict, Any, NoReturn, Union, Tuple
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from . import Instrument
    from jwst.pipeline import JWSTPipeline
    from jwst.datamodels import ImageModel

from .decorators import run_in_self_dir, log_time
from . import config, NIRCam, galfind_logger
from . import useful_funcs_austind as funcs


class Raw_JWST_Data:

    def __init__(
        self: Self,
        survey: str,
        pid: int,
        instrument: Type[Instrument] = NIRCam,
    ):
        if instrument.__name__ != "NIRCam":
            raise ValueError("Raw_Data only currently supports 'NIRCam'")
        self.instrument = instrument()
        self.survey = survey
        self.pid = pid

    @property
    def folder_name(self: Self) -> str:
        return f"{config['DEFAULT']['GALFIND_DATA']}/" + \
            f"{self.instrument.facility.__class__.__name__.lower()}/PID={self.pid}"

    def __repr__(self: Self) -> str:
        return f"Raw_{self.instrument.__class__.__name__}_Data({self.survey},PID={self.pid})"
    
    def __call__(
        self: Self,
        remove_1overf: bool = True,
        subdivide: Optional[str] = None,
        n_cores: int = 1,
        input_crds: int = 1364,
        pre_download_refs: bool = False,
    ):
        if self.instrument.__class__.__name__ == "NIRCam":
            self._call_nircam(
                remove_1overf=remove_1overf,
                subdivide=subdivide,
                n_cores=n_cores,
                input_crds=input_crds,
                pre_download_refs=pre_download_refs,
            )
        else:
            raise NotImplementedError(
                f"{self.instrument.__class__.__name__} is not implemented for Raw_JWST_Data.__call__"
            )
    
    def _call_nircam(
        self: Self,
        remove_1overf: bool = True,
        subdivide: Optional[str] = None,
        n_cores: int = 1,
        input_crds: int = 1364,
        pre_download_refs: bool = False,
    ):
        # download the data from MAST
        self.download()
        self.move_uncals()

        # run the stage 1 pipeline
        stage1_steps = {
            "jump": {
                "expand_large_events": True,
            },
        }
        if remove_1overf:
            stage1_steps = {
                **stage1_steps,
                "clean_flicker_noise": {
                    "skip": False,
                }
            }
        self.run_stage1(
            input_crds = input_crds,
            steps = stage1_steps,
            n_cores = n_cores,
            pre_download_refs = pre_download_refs,
        )

        # run stage 2 of the JWST pipeline
        stage2_steps = {}
        self.run_stage2(
            input_crds = input_crds,
            steps = stage2_steps,
            n_cores = n_cores,
            pre_download_refs = pre_download_refs,
        )
        # run post stage 2 steps - bg subtraction and wisp removal


    @log_time(logging.INFO, u.min)
    @run_in_self_dir(lambda self: self.folder_name)
    def query_mast(
        self: Self
    ) -> List[str]:
        from astroquery.mast import Observations
        instrument_name = f"{self.instrument.__class__.__name__.upper()}/IMAGE"
        obs_table = Observations.query_criteria(
            instrument_name = instrument_name,
            proposal_id = str(self.pid)
        )
        #print(obs_table, obs_table["target_name"], obs_table.colnames)
        data_products = Observations.get_product_list(obs_table)
        # save product list
        filtered_data_products = Observations.filter_products(
            data_products,
            productSubGroupDescription = "UNCAL",
        )

        if filtered_data_products is not None and len(filtered_data_products) != 0:
            # write filtered data products to a file
            save_path = f"{self.instrument.__class__.__name__}_{self.pid}_uncals.fits"
            filtered_data_products.write(save_path, overwrite=True)
            galfind_logger.info(
                f"{len(filtered_data_products)} entries saved to {save_path}"
            )

        manifest = Observations.download_products(
            data_products,
            productSubGroupDescription = "UNCAL",
            curl_flag = True
        )
        self.download_products = manifest["Local Path"].tolist()
        galfind_logger.info(
            f"Queried {self.download_products=} products for {instrument_name=} {self.pid=} from MAST"
        )
        return self.download_products
    
    @log_time(logging.INFO, u.hour)
    @run_in_self_dir(lambda self: f"{self.folder_name}/downloads")
    def download(
        self: Self,
    ):
        if not hasattr(self, "download_products"):
            self.query_mast()
        for input in self.download_products:
            process = subprocess.Popen(["bash", input])
            process.wait()

    @run_in_self_dir(lambda self: self.folder_name)
    def move_uncals(
        self: Self
    ):
        uncals = glob.glob("downloads/*/*/*/*_uncal.fits")
        # move all of these files to an uncals directory
        os.makedirs("uncals", exist_ok=True)
        for file in uncals:
            os.rename(file, f"uncals/{os.path.basename(file)}")

    @staticmethod
    def set_crds_context(input_crds: int = 1364) -> str:
        import crds
        crds_context = f"jwst_{input_crds}.pmap"
        try:
            crds.client.get_reference_names(crds_context)
            galfind_logger.debug(
                f"{crds_context=} is valid and all files are accessible."
            )
            os.environ["CRDS_CONTEXT"] = crds_context
            galfind_logger.info(
                f"Set {crds_context=} for JWST data reduction"
            )
        except crds.exceptions.CrdsError as e:
            galfind_logger.critical(f"{crds_context=} failed certification.")
            galfind_logger.critical("Error:", e)
        return crds_context

    @staticmethod
    def pre_download_refs(
        filenames: Union[NDArray[str], List[str]],
        input_crds: int = 1364,
    ) -> List[str]:
        import crds
        os.environ["CRDS_PATH"] = f"{config['DEFAULT']['GALFIND_DATA']}/crds_cache"
        crds_context = Raw_JWST_Data.set_crds_context(input_crds)
        suffixes = np.unique([file.split("_")[-1].replace(".fits", "") for file in filenames])
        assert len(suffixes) == 1, \
            f"Expected all files to have the same suffix, but found {suffixes=}"
        suffix = suffixes[0]
        [
            crds.getreferences(dict(fits.getheader(file)), context=crds_context)
            for file in tqdm(
                filenames,
                desc=f"Downloading CRDS references to {os.environ['CRDS_PATH']} " + \
                    f"for {suffix} files with {os.environ['CRDS_CONTEXT']=}",
                total=len(filenames),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO
            )
        ]

    @log_time(logging.INFO, u.s)
    @run_in_self_dir(lambda self: self.folder_name)
    def make_asn(
        self: Self,
        split_by: Optional[str, List[str]] = None,
        input_crds: int = 1364,
        hdr_cols: List[str] = [
            "TARG_RA",
            "TARG_DEC",
            "FILTER",
            "OBSERVTN",
            "PROGRAM",
            "TARGPROP",
            "OBS_ID",
        ],
        match_radius: u.Quantity = 25.0 * u.arcmin,
        plot: bool = True,
    ) -> List[str]:
        # set CRDS context
        self.set_crds_context(input_crds)
        cal_filenames = np.array(glob.glob(f"cal_{input_crds}/*_cal.fits"))
        # populate dictionary with relevant header info
        hdr_info = {
            colname: np.full(len(cal_filenames), None, dtype = object)
            for colname in hdr_cols + ["CRVAL1", "CRVAL2"]
        }
        for i, filename in tqdm(
            enumerate(cal_filenames),
            desc = f"Reading {repr(self)} headers",
            total = len(cal_filenames)
        ):
            with fits.open(filename) as hdul:
                hdr = hdul[0].header
                for colname in hdr_cols:
                    hdr_info[colname][i] = hdr.get(colname, "UNKNOWN")
                sci_hdr = hdul["SCI"].header
                for colname in ["CRVAL1", "CRVAL2"]:
                    hdr_info[colname][i] = sci_hdr.get(colname, "UNKNOWN")

        # perform appropriate split
        if split_by is not None:
            if split_by == "sky":
                sky_coords = SkyCoord(
                    np.array(hdr_info["CRVAL1"]).astype(float) * u.deg,
                    np.array(hdr_info["CRVAL2"]).astype(float) * u.deg
                )
                groups = funcs.group_positions(sky_coords, match_radius = match_radius)
                groups = {f"{self.survey}-{name}": filenames for name, filenames in groups.items()}
                plot_subdir = f"sky<{match_radius.to(u.arcmin).value:.1f}arcmin"
            else:
                err_message = f"{split_by=} not in ['sky']"
                galfind_logger.critical(err_message)
                raise ValueError(err_message)
        else:
            groups = {self.survey: cal_filenames}
        
        if plot:
            fig, ax = plt.subplots(figsize = (10, 10))
            # plot all footprints
            all_footprints = funcs.footprints_from_files(cal_filenames)
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("Dec [deg]")
            plt.grid(True)
            ax.invert_xaxis() # RA increases to the left

            for f, coords in all_footprints.items():
                poly = Polygon(coords, closed=False, fill=True, alpha=0.3, facecolor = "grey", edgecolor='k')
                ax.add_patch(poly)

            for group_id, group in groups.items():
                galfind_logger.info(
                    f"Group {group_id} has {len(group)} files"
                )
                footprints = funcs.footprints_from_files(cal_filenames[group])
                added_poly = []
                for f, coords in footprints.items():
                    poly = Polygon(coords, closed=False, fill=True, alpha=0.75, facecolor = "green", edgecolor='k')
                    ax.add_patch(poly)
                    added_poly.append(poly)
                margin = 1.0
                all_coords = np.vstack([poly.get_xy() for poly in added_poly])
                xmin, ymin = all_coords.min(axis=0)
                xmax, ymax = all_coords.max(axis=0)
                dx = (xmax - xmin) * margin
                dy = (ymax - ymin) * margin
                ax.set_xlim(xmin - dx, xmax + dx)
                ax.set_ylim(ymin - dy, ymax + dy)

                save_path = f"{self.folder_name}/asn_{input_crds}/{plot_subdir}/{group_id}.png"
                funcs.make_dirs(save_path)
                plt.savefig(save_path)
                # remove all patches from the axes
                for poly in added_poly:
                    poly.remove()
            plt.close()

        # Generate with explicit ruleset that groups by filter
        for group_id, group in tqdm(
            groups.items(),
            desc = f"Generating associations for {self.survey} {self.pid=}",
            total = len(groups),
            disable = galfind_logger.getEffectiveLevel() > logging.INFO
        ):
            # split by filter
            all_filt_names = np.unique(hdr_info["FILTER"][group])
            for filt in all_filt_names:
                # galfind_logger.info(f"Generating association for {group_id} with {len(group)} files")
                product_name = f"{group_id}-{filt}"
                filt_group = np.array(
                    [id for id in group if hdr_info["FILTER"][id] == filt]
                )
                product_filenames = cal_filenames[filt_group]
                # copy appropriate files to subdirectory
                product_subdir = f"asn_{input_crds}/{group_id}"
                os.makedirs(product_subdir, exist_ok = True)
                # for file in tqdm(
                #     product_filenames,
                #     desc = f"Copying {len(product_filenames)} 'cal' files to {product_subdir}",
                #     total = len(product_filenames),
                #     disable = galfind_logger.getEffectiveLevel() > logging.INFO
                # ):
                #     shutil.copy(
                #         f"{self.folder_name}/{file}",
                #         f"{self.folder_name}/{product_subdir}/{os.path.basename(file)}"
                #     )
                os.system(
                    f"asn_from_list -o {product_subdir}/{filt}.json " + \
                    f"--product-name {product_name} {' '.join([f'{self.folder_name}/{file}' for file in product_filenames])}"
                )
        galfind_logger.info(
            f"Generated associations for {self.survey} {self.pid=} in {self.folder_name}/asn_{input_crds}/"
        )

    @log_time(logging.INFO, u.hour)
    @run_in_self_dir(lambda self: self.folder_name)
    def run_stage1(
        self: Self,
        input_crds: int = 1364,
        steps: Dict[str, Any] = {},
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = "stage1.asdf",
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
    ):
        from jwst.pipeline import Detector1Pipeline
        self.run(
            Detector1Pipeline,
            search_str = "uncals/*_uncal.fits",
            output_suffix = "rate",
            input_crds = input_crds,
            steps = steps,
            config_file = config_file,
            asdf_savename = asdf_savename,
            n_cores = n_cores,
            pre_download_refs = pre_download_refs,
            overwrite = overwrite,
        )

    @log_time(logging.INFO, u.hour)
    @run_in_self_dir(lambda self: self.folder_name)
    def run_stage2(
        self: Self,
        input_crds: int = 1364,
        steps: Dict[str, Any] = {},
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = "stage2.asdf",
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
        wisp_when: str = "pre",
    ):
        #from jwst.pipeline import Image2Pipeline
        if wisp_when == "pre":
            from dewispify.dewisp_stage2 import Image2PipelinePreDewisp as Image2PipelineDewisp
        elif wisp_when == "post":
            from dewispify.dewisp_stage2 import Image2PipelinePostDewisp as Image2PipelineDewisp
        else:
            err_message = f"{wisp_when=} not in ['pre', 'post']"
            galfind_logger.critical(err_message)
            raise ValueError(err_message)
        # ensure steps has a "wisps" entry
        if "wisps" not in steps.keys():
            steps["wisps"] = {}
        if "wisps" in steps:
            steps["wisps"]["wisp_when"] = wisp_when
            steps_wisp_when = steps["wisps"].get("wisp_when", None)
            if steps_wisp_when is not None:
                if steps_wisp_when != wisp_when:
                    galfind_logger.warning(
                        f"Overriding {steps['wisps']['wisp_when']=} with {wisp_when=}"
                    )
        self.run(
            Image2PipelineDewisp,
            search_str = f"rate_{input_crds}/*_rate.fits",
            output_suffix = "cal",
            input_crds = input_crds,
            steps = steps,
            config_file = config_file,
            asdf_savename = asdf_savename,
            n_cores = n_cores,
            pre_download_refs = pre_download_refs,
            overwrite = overwrite,
        )

    @log_time(logging.INFO, u.hour)
    @run_in_self_dir(lambda self: self.folder_name)
    def run_stage3(
        self: Self,
        input_crds: int = 1364,
        steps: Dict[str, Any] = {},
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = "stage3.asdf",
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
    ):
        from jwst.pipeline import Image3Pipeline
        self.run(
            Image3Pipeline,
            search_str = f"asn_{input_crds}/*/*.json",
            output_suffix = "science",
            input_crds = input_crds,
            steps = steps,
            config_file = config_file,
            asdf_savename = asdf_savename,
            n_cores = n_cores,
            pre_download_refs = pre_download_refs,
            overwrite = overwrite,
        )

    @log_time(logging.INFO, u.hour)
    @run_in_self_dir(lambda self: self.folder_name)
    def run(
        self: Self,
        pipe_cls: Type[JWSTPipeline],
        search_str: str,
        output_suffix: str,
        input_crds: int = 1364,
        steps: Dict[str, Any] = {},
        config_file: Optional[str] = None,
        asdf_savename: Optional[str] = None,
        n_cores: int = 1,
        pre_download_refs: bool = False,
        overwrite: bool = False,
    ):
        self.set_crds_context(input_crds)
        os.environ["CRDS_PATH"] = f"{config['DEFAULT']['GALFIND_DATA']}/crds_cache"

        # retrieve all files in this directory
        filenames = glob.glob(search_str)
        if len(filenames) == 0:
            galfind_logger.critical(
                f"No files found in {os.getcwd()}/{search_str}!"
            )
            return
        else:
            galfind_logger.info(
                f"Found {len(filenames)} {repr(self)} " + \
                search_str.split('*')[-1].replace('_', '').replace('.fits', '') + \
                " files for processing!"
            )

        if pre_download_refs:
            self.pre_download_refs(filenames, input_crds=input_crds)

        output_dir = f"{output_suffix}_{input_crds}"
        os.makedirs(output_dir, exist_ok=True)
        # write asdf
        if asdf_savename is not None and not Path(asdf_savename).is_file() or overwrite:
            # make stage 1 pipeline object
            if config_file is not None:
                galfind_logger.info(f"Loading config file: {config_file}")
                assert Path(config_file).is_file(), \
                    galfind_logger.critical(
                        f"{config_file=} does not exist!"
                    )
                if steps != {}:
                    galfind_logger.warning(
                        f"{steps=} ignored when using a config file."
                    )
                pipe = pipe_cls.from_config_file(config_file)
            else:
                pipe = pipe_cls(steps = steps)
            pipe.output_dir = output_dir
            pipe.export_config(asdf_savename)
            galfind_logger.info(
                f"Saved {pipe.__class__.__name__} pipeline configuration for " + \
                f"{self.instrument.__class__.__name__} {self.pid=} " + \
                f"to {os.getcwd()}/{asdf_savename}!"
            )

        if n_cores > 1:
            from multiprocessing import Pool
            stage1_pipe_arr = np.full(len(filenames), pipe_cls)
            steps_arr = np.full(len(filenames), steps)
            output_dir_arr = np.full(len(filenames), output_dir)
            output_suffix_arr = np.full(len(filenames), output_suffix)
            tasks = zip(stage1_pipe_arr, filenames, steps_arr, output_dir_arr, output_suffix_arr)
            
            with Pool(n_cores) as pool:
                outputs = np.full(len(filenames), None, dtype=object)
                for i, (output, err) in tqdm(
                    enumerate(pool.starmap(self._call_stage, tasks)),
                    desc=f"Running {pipe_cls.__name__} " + \
                        f"on files from {self.pid=} with " + \
                        f"{os.environ['CRDS_CONTEXT']=} using {n_cores=}",
                    total=len(filenames),
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO
                ):
                    outputs[i] = output
                    # throw errors after 'processing' all files
                    if err is not None:
                        galfind_logger.error(err)
        else:
            outputs = np.full(len(filenames), None, dtype=object)
            errs = np.full(len(filenames), None, dtype=object)
            for i, filename in tqdm(enumerate(filenames), 
                desc=f"Running stage 1 on uncal files from {self.pid=} with {os.environ['CRDS_CONTEXT']=}",
                total=len(filenames),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO
            ):
                output, err = self._call_stage(pipe_cls, filename, steps, output_dir, output_suffix)
                outputs[i] = output
                errs[i] = err
            # throw errors after 'processing' all files
            for err in errs:
                if err is not None:
                    galfind_logger.error(err)
        return outputs

    @staticmethod
    def _call_stage(
        pipe_cls: Type[JWSTPipeline],
        file: str,
        steps: Dict[str, Any],
        output_dir: str,
        output_suffix: str,
    ) -> Tuple[Optional[ImageModel], Optional[str]]:
        """Run pipeline on a single file."""
        import jwst
        # TODO: Generalize this to work for stage 3 as well!
        input_suffix = file.split("_")[-1]
        out_filename = file.split("/")[-1].replace(input_suffix, output_suffix)
        output = None
        err = None
        if not Path(f"{output_dir}/{out_filename}.fits").is_file():
            try:
                # classmethod
                output = pipe_cls.call(
                    file,
                    steps = steps,
                    output_dir = output_dir,
                    save_results = True
                )
            except Exception as e:
                err = "\n--- ERROR PROCESSING FILE ---\n" + \
                    f"File: {file}\n" + \
                    f"Error Type: {type(e).__name__}\n" + \
                    f"Error Message: {e}\n" + \
                    f"Traceback:\n{traceback.format_exc()}"
        return output, err

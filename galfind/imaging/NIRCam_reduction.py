"""JWST NIRCam raw data reduction and calibration.

Wraps MAST queries and JWST calibration pipeline Stages 1-3 (detector-level,
image-level, and association-based resampling) for NIRCam data.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib.patches import Polygon
from numpy.typing import NDArray
from tqdm import tqdm

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from jwst.datamodels import ImageModel
    from jwst.pipeline import JWSTPipeline

    from . import Instrument

from .. import NIRCam, config, galfind_logger
from ..utils import useful_funcs_austind as funcs
from ..utils.decorators import log_time, run_in_self_dir


class Raw_JWST_Data:
    """Downloads and reduces raw JWST NIRCam imaging data through the
        JWST calibration pipeline.

    Wraps `astroquery` MAST queries/downloads and the ``jwst``
    pipeline's Stage 1-3 processing (detector-level calibration, image
    calibration, and association-based resampling/combination) for a
    single JWST programme ID.

    Parameters
    ----------
    survey : `str`
        Name of the survey this data belongs to.
    pid : `int`
        JWST proposal/programme ID to query and reduce data for.
    instrument : `Type[Instrument]`, optional
        The `Instrument` subclass to reduce data for. Only `NIRCam` is
        currently supported. Default is `NIRCam`.

    Attributes
    ----------
    instrument : `Instrument`
        Instance of the `instrument` class.
    survey : `str`
        Name of the survey.
    pid : `int`
        JWST proposal/programme ID.
    download_products : `list` of `str`
        Local paths to the UNCAL products' download scripts. Only set
        once `query_mast` has been called.
    """

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
        """`str`: Local directory this programme's raw/reduced data is
        stored under.

        `"{GALFIND_DATA}/{facility_name}/PID={pid}"`.
        """
        base_data = config["DEFAULT"]["GALFIND_DATA"]
        facility = self.instrument.facility.__class__.__name__.lower()
        return f"{base_data}/{facility}/PID={self.pid}"

    def __repr__(self: Self) -> str:
        class_name = self.instrument.__class__.__name__
        return f"Raw_{class_name}_Data({self.survey},PID={self.pid})"

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
                f"{self.instrument.__class__.__name__} is not implemented "
                "for Raw_JWST_Data.__call__"
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
                },
            }
        self.run_stage1(
            input_crds=input_crds,
            steps=stage1_steps,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
        )

        # run stage 2 of the JWST pipeline
        stage2_steps = {}
        self.run_stage2(
            input_crds=input_crds,
            steps=stage2_steps,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
        )
        # run post stage 2 steps - bg subtraction and wisp removal

    @log_time(logging.INFO, u.min)
    @run_in_self_dir(lambda self: self.folder_name)
    def query_mast(self: Self) -> List[str]:
        """Query MAST for JWST uncalibrated data products.

        Downloads uncalibrated (UNCAL) raw data from the MAST archive for the
        specified program ID and instrument.

        Returns
        -------
        `list` of `str`
            Local file paths to downloaded UNCAL data products.
        """
        from astroquery.mast import Observations

        instrument_name = f"{self.instrument.__class__.__name__.upper()}/IMAGE"
        obs_table = Observations.query_criteria(
            instrument_name=instrument_name, proposal_id=str(self.pid)
        )
        # print(obs_table, obs_table["target_name"], obs_table.colnames)
        data_products = Observations.get_product_list(obs_table)
        # save product list
        filtered_data_products = Observations.filter_products(
            data_products,
            productSubGroupDescription="UNCAL",
        )

        if (
            filtered_data_products is not None
            and len(filtered_data_products) != 0
        ):
            # write filtered data products to a file
            save_path = (
                f"{self.instrument.__class__.__name__}_{self.pid}_uncals.fits"
            )
            filtered_data_products.write(save_path, overwrite=True)
            galfind_logger.info(
                f"{len(filtered_data_products)} entries saved to {save_path}"
            )

        manifest = Observations.download_products(
            data_products, productSubGroupDescription="UNCAL", curl_flag=True
        )
        self.download_products = manifest["Local Path"].tolist()
        galfind_logger.info(
            f"Queried {self.download_products=} products for "
            f"{instrument_name=} {self.pid=} from MAST"
        )
        return self.download_products

    @log_time(logging.INFO, u.hour)
    @run_in_self_dir(lambda self: f"{self.folder_name}/downloads")
    def download(
        self: Self,
    ) -> None:
        """Download MAST data products using curl scripts.

        Executes curl shell scripts generated by MAST query to download
        raw data.
        """
        if not hasattr(self, "download_products"):
            self.query_mast()
        for input in self.download_products:
            process = subprocess.Popen(["bash", input])
            process.wait()

    @run_in_self_dir(lambda self: self.folder_name)
    def move_uncals(self: Self) -> None:
        """Organize uncalibrated data files into a dedicated directory.

        Moves all UNCAL FITS files from nested download directories into a
        single 'uncals' directory for easier access.
        """
        uncals = glob.glob("downloads/*/*/*/*_uncal.fits")
        # move all of these files to an uncals directory
        os.makedirs("uncals", exist_ok=True)
        for file in uncals:
            os.rename(file, f"uncals/{os.path.basename(file)}")

    @staticmethod
    def set_crds_context(input_crds: int = 1364) -> str:
        """Set the CRDS context for JWST calibration reference data.

        Configures the CRDS (Calibration Reference Data System) context file
        for JWST data reduction pipeline.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS pipeline version number. Default is 1364.

        Returns
        -------
        `str`
            CRDS context file name (e.g., "jwst_1364.pmap").
        """
        import crds

        crds_context = f"jwst_{input_crds}.pmap"
        try:
            crds.client.get_reference_names(crds_context)
            galfind_logger.debug(
                f"{crds_context=} is valid and all files are accessible."
            )
            os.environ["CRDS_CONTEXT"] = crds_context
            galfind_logger.info(f"Set {crds_context=} for JWST data reduction")
        except crds.exceptions.CrdsError as e:
            galfind_logger.critical(f"{crds_context=} failed certification.")
            galfind_logger.critical("Error:", e)
        return crds_context

    @staticmethod
    def pre_download_refs(
        filenames: Union[NDArray[str], List[str]],
        input_crds: int = 1364,
    ) -> List[str]:
        """Pre-download CRDS calibration reference files for JWST data.

        Retrieves all calibration reference files needed for the specified
        UNCAL data files before running the full reduction pipeline.

        Parameters
        ----------
        filenames : `numpy.ndarray` or `list` of `str`
            Paths to UNCAL FITS files requiring calibration references.
        input_crds : `int`, optional
            CRDS pipeline version number. Default is 1364.
        """
        import crds

        os.environ["CRDS_PATH"] = (
            f"{config['DEFAULT']['GALFIND_DATA']}/crds_cache"
        )
        crds_context = Raw_JWST_Data.set_crds_context(input_crds)
        suffixes = np.unique(
            [file.split("_")[-1].replace(".fits", "") for file in filenames]
        )
        assert len(suffixes) == 1, (
            f"Expected all files to have the same suffix, but found "
            f"{suffixes=}"
        )
        suffix = suffixes[0]
        [
            crds.getreferences(
                dict(fits.getheader(file)), context=crds_context
            )
            for file in tqdm(
                filenames,
                desc=(
                    f"Downloading CRDS references to "
                    f"{os.environ['CRDS_PATH']} for {suffix} files with "
                    f"{os.environ['CRDS_CONTEXT']=}"
                ),
                total=len(filenames),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
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
        """Create association (ASN) tables for JWST pipeline processing.

        Groups calibrated data files into associations based on sky position,
        filter, or other header criteria for processing through the JWST
        reduction pipeline.

        Parameters
        ----------
        split_by : `str` or `list` of `str`, optional
            Grouping criterion ("sky" to group by position). Default is None.
        input_crds : `int`, optional
            CRDS pipeline version. Default is 1364.
        hdr_cols : `list` of `str`, optional
            FITS header columns to track. Default includes RA, DEC,
            FILTER, etc.
        match_radius : `astropy.units.Quantity`, optional
            Sky position matching radius. Default is 25 arcmin.
        plot : `bool`, optional
            Whether to generate diagnostic plots. Default is True.

        Returns
        -------
        `list` of `str`
            Paths to created association JSON files.
        """
        # set CRDS context
        self.set_crds_context(input_crds)
        cal_filenames = np.array(glob.glob(f"cal_{input_crds}/*_cal.fits"))
        # populate dictionary with relevant header info
        hdr_info = {
            colname: np.full(len(cal_filenames), None, dtype=object)
            for colname in hdr_cols + ["CRVAL1", "CRVAL2"]
        }
        for i, filename in tqdm(
            enumerate(cal_filenames),
            desc=f"Reading {repr(self)} headers",
            total=len(cal_filenames),
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
                    np.array(hdr_info["CRVAL2"]).astype(float) * u.deg,
                )
                groups = funcs.group_positions(
                    sky_coords, match_radius=match_radius
                )
                groups = {
                    f"{self.survey}-{name}": filenames
                    for name, filenames in groups.items()
                }
                plot_subdir = (
                    f"sky<{match_radius.to(u.arcmin).value:.1f}arcmin"
                )
            else:
                err_message = f"{split_by=} not in ['sky']"
                galfind_logger.critical(err_message)
                raise ValueError(err_message)
        else:
            groups = {self.survey: cal_filenames}

        if plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            # plot all footprints
            all_footprints = funcs.footprints_from_files(cal_filenames)
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("Dec [deg]")
            plt.grid(True)
            ax.invert_xaxis()  # RA increases to the left

            for f, coords in all_footprints.items():
                poly = Polygon(
                    coords,
                    closed=False,
                    fill=True,
                    alpha=0.3,
                    facecolor="grey",
                    edgecolor="k",
                )
                ax.add_patch(poly)

            for group_id, group in groups.items():
                galfind_logger.info(f"Group {group_id} has {len(group)} files")
                footprints = funcs.footprints_from_files(cal_filenames[group])
                added_poly = []
                for f, coords in footprints.items():
                    poly = Polygon(
                        coords,
                        closed=False,
                        fill=True,
                        alpha=0.75,
                        facecolor="green",
                        edgecolor="k",
                    )
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

                plot_dir = f"{self.folder_name}/asn_{input_crds}/{plot_subdir}"
                save_path = f"{plot_dir}/{group_id}.png"
                funcs.make_dirs(save_path)
                plt.savefig(save_path)
                # remove all patches from the axes
                for poly in added_poly:
                    poly.remove()
            plt.close()

        # Generate with explicit ruleset that groups by filter
        for group_id, group in tqdm(
            groups.items(),
            desc=f"Generating associations for {self.survey} {self.pid=}",
            total=len(groups),
            disable=galfind_logger.getEffectiveLevel() > logging.INFO,
        ):
            # split by filter
            all_filt_names = np.unique(hdr_info["FILTER"][group])
            for filt in all_filt_names:
                # galfind_logger.info(
                #     f"Generating association for {group_id} with "
                #     f"{len(group)} files"
                # )
                product_name = f"{group_id}-{filt}"
                filt_group = np.array(
                    [id for id in group if hdr_info["FILTER"][id] == filt]
                )
                product_filenames = cal_filenames[filt_group]
                # copy appropriate files to subdirectory
                product_subdir = f"asn_{input_crds}/{group_id}"
                os.makedirs(product_subdir, exist_ok=True)
                # for file in tqdm(
                #     product_filenames,
                #     desc=(
                #         f"Copying {len(product_filenames)} 'cal' files to "
                #         f"{product_subdir}"
                #     ),
                #     total=len(product_filenames),
                #     disable=galfind_logger.getEffectiveLevel() >
                #         logging.INFO
                # ):
                #     shutil.copy(
                #         f"{self.folder_name}/{file}",
                #         f"{self.folder_name}/{product_subdir}/"
                #         f"{os.path.basename(file)}"
                #     )
                file_paths = [
                    f"{self.folder_name}/{file}" for file in product_filenames
                ]
                file_list = " ".join(file_paths)
                os.system(
                    f"asn_from_list -o {product_subdir}/{filt}.json "
                    f"--product-name {product_name} {file_list}"
                )
        asn_dir = f"{self.folder_name}/asn_{input_crds}/"
        galfind_logger.info(
            f"Generated associations for {self.survey} {self.pid=} in "
            f"{asn_dir}"
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
    ) -> None:
        """Run JWST stage 1 detector calibration pipeline.

        Applies detector-level corrections (bias, dark, nonlinearity) to
        uncalibrated raw data, producing rate images.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version. Default is 1364.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is "stage1.asdf".
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        """
        from jwst.pipeline import Detector1Pipeline

        self.run(
            Detector1Pipeline,
            search_str="uncals/*_uncal.fits",
            output_suffix="rate",
            input_crds=input_crds,
            steps=steps,
            config_file=config_file,
            asdf_savename=asdf_savename,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
            overwrite=overwrite,
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
    ) -> None:
        """Run JWST stage 2 image calibration pipeline.

        Applies image-level corrections (flat-field, photometry, WCS) with
        optional
        wavefront sensing and phase retrieval (WISP) processing.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version. Default is 1364.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is "stage2.asdf".
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        wisp_when : `str`, optional
            When to apply WISP processing ("pre" or "post"). Default is "pre".
        """
        # from jwst.pipeline import Image2Pipeline
        if wisp_when == "pre":
            from dewispify.dewisp_stage2 import (
                Image2PipelinePreDewisp as Image2PipelineDewisp,
            )
        elif wisp_when == "post":
            from dewispify.dewisp_stage2 import (
                Image2PipelinePostDewisp as Image2PipelineDewisp,
            )
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
                        f"Overriding {steps['wisps']['wisp_when']=} with "
                        f"{wisp_when=}"
                    )
        self.run(
            Image2PipelineDewisp,
            search_str=f"rate_{input_crds}/*_rate.fits",
            output_suffix="cal",
            input_crds=input_crds,
            steps=steps,
            config_file=config_file,
            asdf_savename=asdf_savename,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
            overwrite=overwrite,
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
    ) -> None:
        """Run JWST stage 3 image processing pipeline.

        Performs image alignment, astrometric refinement, image stack creation,
        and source catalog extraction for multi-exposure datasets.

        Parameters
        ----------
        input_crds : `int`, optional
            CRDS context version. Default is 1364.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is "stage3.asdf".
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        """
        from jwst.pipeline import Image3Pipeline

        self.run(
            Image3Pipeline,
            search_str=f"asn_{input_crds}/*/*.json",
            output_suffix="science",
            input_crds=input_crds,
            steps=steps,
            config_file=config_file,
            asdf_savename=asdf_savename,
            n_cores=n_cores,
            pre_download_refs=pre_download_refs,
            overwrite=overwrite,
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
    ) -> None:
        """Execute a JWST pipeline stage on input data files.

        Generic pipeline runner that applies the specified JWST pipeline to
        matching input files with optional parallel processing.

        Parameters
        ----------
        pipe_cls : `Type[JWSTPipeline]`
            JWST pipeline class (Detector1Pipeline, Image2Pipeline, etc.).
        search_str : `str`
            Glob pattern to match input files.
        output_suffix : `str`
            Suffix for output files (e.g., "rate", "cal", "science").
        input_crds : `int`, optional
            CRDS context version. Default is 1364.
        steps : `dict`, optional
            Pipeline step parameters. Default is empty dict.
        config_file : `str`, optional
            Path to pipeline config file. Default is None.
        asdf_savename : `str`, optional
            Name for saved pipeline state. Default is None.
        n_cores : `int`, optional
            Number of CPU cores to use. Default is 1.
        pre_download_refs : `bool`, optional
            Whether to pre-download calibration references. Default is False.
        overwrite : `bool`, optional
            Whether to overwrite existing output. Default is False.
        """
        self.set_crds_context(input_crds)
        os.environ["CRDS_PATH"] = (
            f"{config['DEFAULT']['GALFIND_DATA']}/crds_cache"
        )

        # retrieve all files in this directory
        filenames = glob.glob(search_str)
        if len(filenames) == 0:
            galfind_logger.critical(
                f"No files found in {os.getcwd()}/{search_str}!"
            )
            return
        else:
            galfind_logger.info(
                f"Found {len(filenames)} {repr(self)} "
                + search_str.split("*")[-1]
                .replace("_", "")
                .replace(".fits", "")
                + " files for processing!"
            )

        if pre_download_refs:
            self.pre_download_refs(filenames, input_crds=input_crds)

        output_dir = f"{output_suffix}_{input_crds}"
        os.makedirs(output_dir, exist_ok=True)
        # write asdf
        if (
            asdf_savename is not None
            and not Path(asdf_savename).is_file()
            or overwrite
        ):
            # make stage 1 pipeline object
            if config_file is not None:
                galfind_logger.info(f"Loading config file: {config_file}")
                assert Path(config_file).is_file(), galfind_logger.critical(
                    f"{config_file=} does not exist!"
                )
                if steps != {}:
                    galfind_logger.warning(
                        f"{steps=} ignored when using a config file."
                    )
                pipe = pipe_cls.from_config_file(config_file)
            else:
                pipe = pipe_cls(steps=steps)
            pipe.output_dir = output_dir
            pipe.export_config(asdf_savename)
            galfind_logger.info(
                f"Saved {pipe.__class__.__name__} pipeline configuration for "
                + f"{self.instrument.__class__.__name__} {self.pid=} "
                + f"to {os.getcwd()}/{asdf_savename}!"
            )

        if n_cores > 1:
            from multiprocessing import Pool

            stage1_pipe_arr = np.full(len(filenames), pipe_cls)
            steps_arr = np.full(len(filenames), steps)
            output_dir_arr = np.full(len(filenames), output_dir)
            output_suffix_arr = np.full(len(filenames), output_suffix)
            tasks = zip(
                stage1_pipe_arr,
                filenames,
                steps_arr,
                output_dir_arr,
                output_suffix_arr,
            )

            with Pool(n_cores) as pool:
                outputs = np.full(len(filenames), None, dtype=object)
                for i, (output, err) in tqdm(
                    enumerate(pool.starmap(self._call_stage, tasks)),
                    desc=f"Running {pipe_cls.__name__} "
                    + f"on files from {self.pid=} with "
                    + f"{os.environ['CRDS_CONTEXT']=} using {n_cores=}",
                    total=len(filenames),
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                ):
                    outputs[i] = output
                    # throw errors after 'processing' all files
                    if err is not None:
                        galfind_logger.error(err)
        else:
            outputs = np.full(len(filenames), None, dtype=object)
            errs = np.full(len(filenames), None, dtype=object)
            for i, filename in tqdm(
                enumerate(filenames),
                desc=(
                    f"Running stage 1 on uncal files from {self.pid=} with "
                    f"{os.environ['CRDS_CONTEXT']=}"
                ),
                total=len(filenames),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            ):
                output, err = self._call_stage(
                    pipe_cls, filename, steps, output_dir, output_suffix
                )
                outputs[i] = output
                errs[i] = err
            # throw errors after 'processing' all files
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
        # TODO: Generalize this to work for stage 3 as well!
        input_suffix = file.split("_")[-1]
        out_filename = file.split("/")[-1].replace(input_suffix, output_suffix)
        output = None
        err = None
        if not Path(f"{output_dir}/{out_filename}.fits").is_file():
            try:
                # classmethod
                output = pipe_cls.call(
                    file, steps=steps, output_dir=output_dir, save_results=True
                )
            except Exception as e:
                err = (
                    "\n--- ERROR PROCESSING FILE ---\n"
                    + f"File: {file}\n"
                    + f"Error Type: {type(e).__name__}\n"
                    + f"Error Message: {e}\n"
                    + f"Traceback:\n{traceback.format_exc()}"
                )
        return output, err

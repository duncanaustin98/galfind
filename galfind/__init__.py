"""GALFIND: A comprehensive galaxy photometry and SED fitting framework.

GALFIND provides tools for galaxy photometry, SED fitting, morphological analysis,
and statistical characterization, with deep integrations for JWST/HST data processing
and multiple external SED fitting codes (EAZY, LePhare, Bagpipes).

Key components:
    - Galaxy: Represents individual sources with photometry and SED fits
    - Catalogue: Collection of Galaxy objects with FITS I/O
    - Data, Filter, Instrument: Imaging data and instrumental setup
    - Photometry, SED, SED_result: Photometry and SED fit storage
    - Selector: Galaxy sample selection with customizable criteria
    - Property_calculator: Derived property computation with uncertainties
    - Morphology, Spectrum: Morphological and spectroscopic data
"""

# __init__.py

import time
start = time.time()
import os
import configparser
import json
import logging
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
end = time.time()
#print(f"__init__ imports took {end - start}s")

galfind_dir = "/".join(__file__.split("/")[:-1])
try:
    config_dir = os.environ['GALFIND_CONFIG_DIR']
except:
    config_dir = f"{galfind_dir}/../configs"

try:
    config_path = f"{config_dir}/{os.environ['GALFIND_CONFIG_NAME']}"
except KeyError:
    config_path = f"{config_dir}/galfind_config.ini"

print("Reading GALFIND config file from:", config_path)

# note whether the __init__ is running in a workflow
in_workflow = os.environ.get("GITHUB_ACTIONS") == "true"

# configuration variables
config = configparser.ConfigParser()
config.read(config_path)
config.set("DEFAULT", "GALFIND_DIR", galfind_dir)
config.set("DEFAULT", "CONFIG_DIR",  f"{galfind_dir}/../configs")

# on ReadTheDocs, override machine-specific paths with writable defaults
if os.environ.get("READTHEDOCS") == "True":
    config.set("DEFAULT", "GALFIND_WORK", "/tmp/galfind_docs_build")
    config.set("DEFAULT", "GALFIND_DATA", "/tmp/galfind_docs_build")

# resolve to absolute paths so downstream-derived paths (which lean on
# these via configparser interpolation) still work once code temporarily
# os.chdir()s elsewhere (e.g. run_in_dir-decorated SExtractor/EAZY calls)
for _root_key in ("GALFIND_WORK", "GALFIND_DATA"):
    config.set("DEFAULT", _root_key, os.path.abspath(config["DEFAULT"][_root_key]))

# Make IS_CLUSTER variable from the config parameters
if config["DEFAULT"]["SURVEY"] in json.loads(config.get("Other", "CLUSTER_FIELDS")):
    config.set("DEFAULT", "IS_CLUSTER", "YES")
else:
    config.set("DEFAULT", "IS_CLUSTER", "NO")

# set up logging
if config.getboolean("DEFAULT", "USE_LOGGING"):
    logging.basicConfig(
        level={
            "NOTSET": logging.NOTSET,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }[config["DEFAULT"]["LOGGING_LEVEL"]]
    )
    # Create a logger instance
    galfind_logger = logging.getLogger(__name__)
    # don't add file handler to galfind_logger if in workflow
    if not in_workflow:
        current_timestamp = time.strftime("%Y-%m-%d", time.gmtime())
        log_file_name = f"{current_timestamp}.log"
        os.makedirs(config['DEFAULT']['LOGGING_OUT_DIR'], exist_ok = True) # make directory if it doesnt already exist
        log_file_path = f"{config['DEFAULT']['LOGGING_OUT_DIR']}/{log_file_name}"
        # Create a file handler
        file_handler = logging.FileHandler(log_file_path)
        #file_handler.setLevel()
        galfind_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(galfind_log_formatter)
        galfind_logger.addHandler(file_handler)
        try:
            os.chmod(log_file_path, 0o777)
        except PermissionError:
            galfind_logger.debug(f"Could not change permissions of {log_file_path} to 777.")
    # print out the default galfind config file parameters
    # for i, (option, value) in enumerate(config["DEFAULT"].items()):
    #     if i == 0:
    #         # Temporarily remove the formatter
    #         galfind_logger.handlers[0].setFormatter(logging.Formatter(''))
    #         galfind_logger.info(f"{config_path.split('/')[-1]}: [DEFAULT]")
    #         galfind_logger.info("------------------------------------------")
    #         # Reattach the original formatter
    #         galfind_logger.handlers[0].setFormatter(galfind_log_formatter)
    #     galfind_logger.info(f"{option}: {value}")
    # for section in config.sections():
    #     galfind_logger.handlers[0].setFormatter(logging.Formatter(''))
    #     galfind_logger.info(f"{config_path.split('/')[-1]}: [{section}]")
    #     galfind_logger.info("------------------------------------------")
    #     galfind_logger.handlers[0].setFormatter(galfind_log_formatter)
    #     for option in config.options(section):
    #         if option not in config["DEFAULT"].keys():
    #             value = config.get(section, option)
    #             galfind_logger.info(f"{option}: {value}")
    # # Temporarily remove the formatter
    # galfind_logger.handlers[0].setFormatter(logging.Formatter(''))
    # galfind_logger.info("------------------------------------------")
    # Reattach the original formatter
        galfind_logger.handlers[0].setFormatter(galfind_log_formatter)
else:
    raise (Exception("galfind currently not set up to allow users to ignore logging!"))

# limit number of threads to N_CORES
n_threads = str(config.getint("DEFAULT", "N_CORES"))
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads
os.environ["OMP_NUM_THREADS"] = n_threads

try:
    import mkl
    mkl.set_num_threads(int(n_threads))
except:
    galfind_logger.debug(f"Failed to set mkl.set_num_threads to {n_threads}.")

# set cosmology
astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.725)

# set lyman limit and lyman alpha wavelengths
wav_lyman_lim = 911.8  # * u.AA

from .utils import useful_funcs_austind
from .utils import utils
from .utils import exceptions
from .visualization import figs
from .utils import decorators

# all_filt_names will be computed at the end of this file after all imports

# Package-level exports:
# Import classes from their respective subpackages, e.g.:
#   from galfind.catalogues import Catalogue
#   from galfind.sed_fitting import EAZY, SED_code
#   from galfind.selection import Redshift_Bin_Selector
__all__ = [
    "config",
    "galfind_logger",
    "astropy_cosmo",
    "wav_lyman_lim",
    "all_filt_names",
    "useful_funcs_austind",
    "utils",
    "exceptions",
    "figs",
    "decorators",
]

# Compute all_filt_names from available Instrument subclasses (after all imports)
from .imaging.Instrument import Instrument
all_filt_names = []
for instr_cls in useful_funcs_austind.all_subclasses(Instrument):
    try:
        instr_inst = instr_cls()
        if hasattr(instr_inst, 'filt_names'):
            all_filt_names.extend(instr_inst.filt_names)
    except Exception:
        pass
all_filt_names = list(set(all_filt_names))

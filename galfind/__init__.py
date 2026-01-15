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
if "hostedtoolcache" in galfind_dir:
    in_workflow = True
else:
    in_workflow = False

# configuration variables
config = configparser.ConfigParser()
config.read(config_path)
config.set("DEFAULT", "GALFIND_DIR", galfind_dir)
config.set("DEFAULT", "CONFIG_DIR",  f"{galfind_dir}/../configs")

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

from . import useful_funcs_austind
from . import figs
from . import decorators
from . import SExtractor, Masking, Depths

from .PSF import PSF_Base, PSF_Cutout
from .Instrument import (
    Facility, JWST, HST, Paranal, Spitzer, Euclid, CFHT, Subaru,
    Instrument, ACS_SBC, ACS_WFC, WFC3_IR, NIRCam, MIRI, VISTA, NISP, VIS, IRAC, MegaCam, HSC
)
instr_to_name_dict = {name: globals()[name]() for name in json.loads(config.get("Other", "INSTRUMENT_NAMES"))}
from .Filter import Filter, Multiple_Filter, Tophat_Filter, U, V, J

# sort bands blue -> red based on central wavelength
all_band_names = [filt.band_name for filt in sorted(Multiple_Filter.from_instruments \
    (list(json.loads(config.get("Other", "INSTRUMENT_NAMES")))), \
    key=lambda band: band.WavelengthCen.to(u.AA).value)]
config.set("Other", "ALL_BANDS", json.dumps(all_band_names))

from .PDF import PDF, SED_fit_PDF, Redshift_PDF, PDF_nD

from .Data import Band_Data_Base, Band_Data, Stacked_Band_Data, Data
from .Cutout import Cutout_Base, Band_Cutout, Band_Cutout_Base, Stacked_Band_Cutout, RGB, Stacked_RGB, Multiple_Band_Cutout, Multiple_RGB, Catalogue_Cutouts

from .Photometry import Photometry, Multiple_Photometry, Mock_Photometry
from .Photometry_obs import Photometry_obs, Multiple_Photometry_obs
from .Photometry_rest import Photometry_rest
from .SED_result import SED_result, Galaxy_SED_results, Catalogue_SED_results

from .SED_codes import SED_code
from .LePhare import LePhare
from .EAZY import EAZY # Failed to `import dust_attenuation`
from .Bagpipes import Bagpipes
from .Brown_Dwarf_Fitter import Template_Fitter, Brown_Dwarf_Fitter

# don't do Bagpipes or LePhare for now
# sed_code_to_name_dict = {
#     sed_code_name: globals()[sed_code_name]()
#     for sed_code_name in [subcls.__name__ for subcls in SED_code.__subclasses__()]
#     if sed_code_name not in ["LePhare", "Bagpipes"]
# }

from .Galaxy import Galaxy

from .Catalogue_Base import Catalogue_Base
from .Multiple_Catalogue import Combined_Catalogue
#from .Multiple_Data import Multiple_Data
from .Catalogue import Catalogue, Catalogue_Creator
from .SED import SED, SED_rest, SED_obs, Mock_SED_rest, Mock_SED_obs
from .SED import (
    Mock_SED_template_set,
    Mock_SED_rest_template_set,
    Mock_SED_obs_template_set,
)

from .Selector import (
    Selector, 
    ID_Selector,
    Multiple_Selector,
    Data_Selector,
    Photometry_Selector,
    SED_fit_Selector,
    Morphology_Selector,
    Region_Selector,
    Ds9_Region_Selector,
    Depth_Region_Selector,
    Multiple_Data_Selector,
    Multiple_Photometry_Selector,
    Multiple_SED_fit_Selector,
    Unmasked_Band_Selector, 
    Unmasked_Bands_Selector, 
    Unmasked_Instrument_Selector,
    Min_Band_Selector,
    Min_Unmasked_Band_Selector,
    Min_Instrument_Unmasked_Band_Selector,
    Mask_Selector,
    Multiple_Mask_Selector,
    Sextractor_Band_Radius_Selector,
    Sextractor_Bands_Radius_Selector,
    Sextractor_Instrument_Radius_Selector,
    Band_SNR_Selector,
    Colour_Selector, 
    Kokorev24_LRD_red1_Selector, 
    Kokorev24_LRD_red2_Selector, 
    Kokorev24_LRD_Selector,
    Bluewards_Lya_Non_Detect_Selector,
    Bluewards_LyLim_Non_Detect_Selector,
    Redwards_Lya_Detect_Selector,
    Lya_Band_Selector,
    Chi_Sq_Lim_Selector,
    Chi_Sq_Diff_Selector,
    Robust_zPDF_Selector,
    Re_Selector,
    EPOCHS_Selector,
    Redshift_Limit_Selector,
    Redshift_Bin_Selector,
    Rest_Frame_Property_Limit_Selector,
    Rest_Frame_Property_Bin_Selector,
    Rest_Frame_Property_Kwarg_Selector,
    Brown_Dwarf_Selector,
    Hainline24_TY_Brown_Dwarf_Selector_1,
    Hainline24_TY_Brown_Dwarf_Selector_2,
)

from .Emission_lines import Emission_line, wav_lyman_alpha, line_diagnostics
from . import IGM_attenuation
from . import lyman_alpha_damping_wing
from .DLA import DLA
from .Dust_Attenuation import Dust_Law, Calzetti00, SMC, Reddy15, Salim18, Modified_Calzetti00, Power_Law_Dust, M99, Reddy15, Reddy18, AUV_from_beta
from .Spectrum import (
    Spectral_Catalogue,
    Spectrum,
    NIRSpec,
    Spectral_Instrument,
    Spectral_Filter,
    Spectral_Grating,
)
from .MCMC import Prior, Flat_Prior, Priors, MCMC_Fitter, Schechter_Mag_Fitter, Schechter_Lum_Fitter, Linear_Fitter, Power_Law_Fitter, Scattered_Linear_Fitter
from .Number_Density_Function import (
    Base_Number_Density_Function,
    Number_Density_Function,
)  # UVLFs, mass functions, etc

from .Property_calculator import (
    Property_Calculator_Base, 
    Property_Calculator,
    Photometry_Property_Loader,
    Band_SNR_Loader,
    Redshift_Extractor,
    Ext_Src_Property_Calculator, 
    Custom_SED_Property_Extractor,
    Custom_Morphology_Property_Extractor,
    Property_Multiplier,
    Property_Divider,
    Re_kpc_Calculator,
    Surface_Density_Calculator,
)

from .Rest_frame_properties import (
    Rest_Frame_Property_Calculator,
    UV_Beta_Calculator,
    UV_Dust_Attenuation_Calculator,
    mUV_Calculator,
    MUV_Calculator,
    LUV_Calculator,
    SFR_UV_Calculator,
    Fesc_From_Beta_Calculator,
    Optical_Continuum_Calculator,
    Optical_Line_EW_Calculator,
    Dust_Attenuation_From_UV_Calculator,
    Line_Dust_Attenuation_From_UV_Calculator,
    Optical_Line_Flux_Calculator,
    Optical_Line_Luminosity_Calculator,
    Ndot_Ion_Calculator,
    Xi_Ion_Calculator,
    SFR_Halpha_Calculator,
)

from .Morphology import Morphology_Result, Morphology_Fitter, Galfit_Fitter
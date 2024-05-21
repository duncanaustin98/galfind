#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:55:23 2023

@author: austind
"""

#__init__.py
from __future__ import absolute_import
import configparser
import json
import logging
import time
import os
import astropy.units as u
import numpy as np
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM

galfind_dir = "/".join(__file__.split("/")[:-1]) 

try:
    config_path = os.environ['GALFIND_CONFIG_PATH']

except KeyError:
    config_path = f"{galfind_dir}/configs/galfind_config.ini" # needs to be able to be changed by the user

print('Reading GALFIND config file from:', config_path)
# configuration variables
config = configparser.ConfigParser()
config.read(config_path)
config.set("DEFAULT", "GALFIND_DIR", galfind_dir)
    
# Make IS_CLUSTER variable from the config parameters
if config["DEFAULT"]["SURVEY"] in json.loads(config.get("Other", "CLUSTER_FIELDS")):
    config.set("DEFAULT", "IS_CLUSTER", "YES")
else:
    config.set("DEFAULT", "IS_CLUSTER", "NO")

# set up logging
if config.getboolean("DEFAULT", "USE_LOGGING"):
    logging.basicConfig(level = {'NOTSET': logging.NOTSET, 'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, \
        'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL}[config["DEFAULT"]["LOGGING_LEVEL"]])
    # Create a logger instance
    galfind_logger = logging.getLogger(__name__)
    #current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    #log_file_name = f"{current_timestamp}.log"
    log_file_name = f"{config['DEFAULT']['SURVEY']}_{config['DEFAULT']['VERSION']}.log"
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
        galfind_logger.warning(f"Could not change permissions of {log_file_path} to 777.")
    # print out the default galfind config file parameters
    for i, (option, value) in enumerate(config["DEFAULT"].items()):
        if i == 0:
            # Temporarily remove the formatter
            galfind_logger.handlers[0].setFormatter(logging.Formatter(''))
            galfind_logger.info(f"{config_path.split('/')[-1]}: [DEFAULT]")
            galfind_logger.info("------------------------------------------")
            # Reattach the original formatter
            galfind_logger.handlers[0].setFormatter(galfind_log_formatter)
        galfind_logger.info(f"{option}: {value}")
    for section in config.sections():
        galfind_logger.handlers[0].setFormatter(logging.Formatter(''))
        galfind_logger.info(f"{config_path.split('/')[-1]}: [{section}]")
        galfind_logger.info("------------------------------------------")
        galfind_logger.handlers[0].setFormatter(galfind_log_formatter)
        for option in config.options(section):
            if option not in config["DEFAULT"].keys():
                value = config.get(section, option)
                galfind_logger.info(f"{option}: {value}")
    # Temporarily remove the formatter
    galfind_logger.handlers[0].setFormatter(logging.Formatter(''))
    galfind_logger.info("------------------------------------------")
    # Reattach the original formatter
    galfind_logger.handlers[0].setFormatter(galfind_log_formatter)
else:
    raise(Exception("galfind currently not set up to allow users to ignore logging!"))

# set cosmology
astropy_cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725)

# set lyman limit and lyman alpha wavelengths
wav_lyman_lim = 911.8 # * u.AA

from . import useful_funcs_austind
from . import NIRCam_aperture_corrections as NIRCam_aper_corr
from . import Depths
from .PDF import PDF, Redshift_PDF, PDF_nD
from .Filter import Filter
from .Data import Data
from .Instrument import Instrument, ACS_WFC, WFC3_IR, NIRCam, MIRI, Combined_Instrument
from .Photometry import Photometry, Multiple_Photometry, Mock_Photometry
from .Photometry_obs import Photometry_obs, Multiple_Photometry_obs
from .Photometry_rest import Photometry_rest
from .SED_result import SED_result, Galaxy_SED_results, Catalogue_SED_results
from .SED_codes import SED_code
from .Multiple_Catalogue import Multiple_Catalogue
from .Multiple_Data import Multiple_Data
from .Catalogue_Base import Catalogue_Base
from .Catalogue import Catalogue
from .Catalogue_Creator import Catalogue_Creator, GALFIND_Catalogue_Creator
from .SED import SED, SED_rest, SED_obs, Mock_SED_rest, Mock_SED_obs
from .SED import Mock_SED_template_set, Mock_SED_rest_template_set, Mock_SED_obs_template_set
from .LePhare import LePhare
from .EAZY import EAZY
from .Bagpipes import Bagpipes
from .Galaxy import Galaxy, Multiple_Galaxy
from .Simulated_Galaxy import Simulated_Galaxy
from .Simulated_Catalogue import Simulated_Catalogue
from . import decorators
from .Emission_lines import Emission_line, wav_lyman_alpha, line_diagnostics
from . import IGM_attenuation
from . import lyman_alpha_damping_wing
from .DLA import DLA
from .Dust_Attenuation import Dust_Attenuation, C00
from .Spectrum import Spectral_Catalogue, Spectrum, NIRSpec, Spectral_Instrument, Spectral_Filter, Spectral_Grating

# ordered band names from blue -> red
config.set("Other", "ALL_BANDS", json.dumps(["F435W","FR459M","F475W","F550M","F555W","F606W","F625W","FR647M","F070W","F775W","F814W","F850LP",
             "F090W","FR914M","F098M","F105W","F110W","F115W","F125W","F127M","F139M","F140W","F140M","F150W","F153M","F160W","F162M","F182M",
             "F200W","F210M","F250M","F277W","F300M","F335M","F356W","F360M","F410M","F430M","F444W","F460M","F480M","F560W","F770W","F1000W","F1130W","F1280W","F1500W","F1800W","F2100W","F2550W"]))
# print(np.array([getattr(globals()[subcls.__class__.__name__], subcls.__class__.__name__)() for subcls in Instrument.__subclasses__()]).flatten())
# config.set("Other", "ALL_BANDS", json.dumps(np.array([subcls().band_names for subcls in Instrument.__subclasses__]).flatten()))

# dynamically add Galaxy selection methods to Catalogue class?


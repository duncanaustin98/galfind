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
config_path = f"{galfind_dir}/configs/galfind_config.ini" # needs to be able to be changed by the user
# configuration variables
config = configparser.ConfigParser()
config.read(config_path)
config.set("DEFAULT", "GALFIND_DIR", galfind_dir)
    
# Make IS_CLUSTER variable from the config parameters
if config["DEFAULT"]["SURVEY"] in json.loads(config.get("Other", "CLUSTER_FIELDS")):
    config.set("DEFAULT", "IS_CLUSTER", "YES")
else:
    config.set("DEFAULT", "IS_CLUSTER", "NO")

# Not currently including all ACS/MIRI bands (does include all NIRCam Wide/Medium band filters), and none are included from WFC3IR yet
config.set("Other", "ALL_BANDS", json.dumps(["f435W","fr459M","f475W","f550M","f555W","f606W","f625W","fr647M","f070W","f775W","f814W","f850LP",
             "f090W","fr914M","f098M","f105W","f110W","f115W","f125W","f127M","f139M","f140W","f140M","f150W","f153M","f160W","f162M","f182M",
             "f200W","f210M","f250M","f277W","f300M","f335M","f356W","f360M","f410M","f430M","f444W","f460M","f480M"]))

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

# set lyman limit wavelength
wav_lyman_lim = 911.8 # * u.AA

from . import useful_funcs_austind
from . import NIRCam_aperture_corrections as NIRCam_aper_corr
from .Data import Data
from .Instrument import Instrument, ACS_WFC, WFC3_IR, NIRCam, MIRI, Combined_Instrument
from .Photometry import Photometry, Multiple_Photometry, Mock_Photometry
from .Photometry_obs import Photometry_obs, Multiple_Photometry_obs
from .Photometry_rest import Photometry_rest
from .SED_result import SED_result, Galaxy_SED_results, Catalogue_SED_results
from .SED_codes import SED_code
from .Catalogue_Base import Catalogue_Base
from .Catalogue import Catalogue
from .Catalogue_Creator import Catalogue_Creator, GALFIND_Catalogue_Creator
from .LePhare import LePhare
from .EAZY import EAZY
from .Bagpipes import Bagpipes
from .Galaxy import Galaxy, Multiple_Galaxy
from .Simulated_Galaxy import Simulated_Galaxy
from .Simulated_Catalogue import Simulated_Catalogue
from . import decorators
from .SED import SED, SED_rest, SED_obs, Mock_SED_rest, Mock_SED_obs
from . import IGM_attenuation

# make IGM grid if it doesn't exist, else load it
if not Path(f"{config['MockSEDs']['IGM_DIR']}/{config['MockSEDs']['IGM_PRESCRIPTION']}_IGM_grid.h5").is_file():
    IGM_grid = IGM_attenuation.make_IGM_transmission_grid(np.linspace(wav_lyman_lim, 1216., 10000), np.linspace(0., 10., 1000))
IGM_transmission_grid, IGM_z_arr, IGM_wav_rest_arr = IGM_attenuation.load_IGM_transmission_grid()



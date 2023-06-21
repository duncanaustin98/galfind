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

config_path = "/nvme/scratch/work/austind/GALFIND/galfind_config.ini" # needs to be able to be changed by the user
# configuration variables
config = configparser.ConfigParser()
config.read(config_path)
config.set("DEFAULT", "GALFIND_DIR", "/".join(__file__.split("/")[:-1]))

# override VERSION variable from the config parameters if required (NOT GENERAL!)
if config["DataReduction"].getint("NIRCAM_PMAP") == 1084:
    if config["DataReduction"]["NIRCAM_PIPELINE_VERSION"] == "1.8.2":
        config.set("DEFAULT", "VERSION", "v8a")
    
# Make IS_CLUSTER variable from the config parameters
if config["DEFAULT"]["SURVEY"] in json.loads(config.get("Other", "CLUSTER_FIELDS")):
    config.set("DEFAULT", "IS_CLUSTER", "YES")
else:
    config.set("DEFAULT", "IS_CLUSTER", "NO")

from . import NIRCam_aperture_corrections as NIRCam_aper_corr
from .Data import Data
from .Instrument import Instrument, ACS, NIRCam, MIRI, Combined_Instrument
from .Catalogue import Catalogue
from .SED_codes import SED_code
from .LePhare import LePhare
from .EAZY import EAZY
from .Bagpipes import Bagpipes
from .Simulated_Catalogue import Simulated_Catalogue
from .Simulated_Galaxy import Simulated_Galaxy
from . import useful_funcs_austind
from . import decorators

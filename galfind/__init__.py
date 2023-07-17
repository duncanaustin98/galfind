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
from astropy.cosmology import FlatLambdaCDM

galfind_dir = "/".join(__file__.split("/")[:-1])
config_path = f"{galfind_dir}/configs/galfind_config.ini" # needs to be able to be changed by the user
# configuration variables
config = configparser.ConfigParser()
config.read(config_path)
config.set("DEFAULT", "GALFIND_DIR", galfind_dir)

# override VERSION variable from the config parameters if required (NOT GENERAL!)
if config["DataReduction"].getint("NIRCAM_PMAP") == 1084:
    if config["DataReduction"]["NIRCAM_PIPELINE_VERSION"] == "1.8.2":
        config.set("DEFAULT", "VERSION", "v8a")
    
# Make IS_CLUSTER variable from the config parameters
if config["DEFAULT"]["SURVEY"] in json.loads(config.get("Other", "CLUSTER_FIELDS")):
    config.set("DEFAULT", "IS_CLUSTER", "YES")
else:
    config.set("DEFAULT", "IS_CLUSTER", "NO")

# Not currently including all ACS/MIRI bands (does include all NIRCam Wide/Medium band filters), and none are included from WFC3IR yet
config.set("Other", "ALL_BANDS", ', '.join(["f435W","fr459M","f475W","f550M","f555W","f606W","f625W","fr647M","f070W","f775W","f814W","f850LP",
             "f090W","f098M","fr914M","f105W","f110W","f115W","f125W","f127M","f139M","f140W","f140M","f150W","f153M","f160W","f162M","f182M",
             "f200W","f210M","f250M","f277W","f300M","f335M","f356W","f360M","f410M","f430M","f444W","f460M","f480M"])) #, "f560W",
             #"f770W", "f1000W","f1130W", "f1280W", "f1500W", "f1800W", "f2100W", "f2550W"]))

# set cosmology
astropy_cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3, Ob0 = 0.05, Tcmb0=2.725)

from . import NIRCam_aperture_corrections as NIRCam_aper_corr
from .Data import Data
from .Instrument import Instrument, ACS_WFC,WFC3IR, NIRCam, MIRI, Combined_Instrument
from .Photometry import Photometry
from .Photometry_obs import Photometry_obs
from .Photometry_rest import Photometry_rest
from .SED_codes import SED_code, SED_result
from .Catalogue import Catalogue
from .Catalogue_Creator import Catalogue_Creator, GALFIND_Catalogue_Creator
from .LePhare import LePhare
from .EAZY import EAZY
from .Bagpipes import Bagpipes
from .Galaxy import Galaxy
from .Simulated_Galaxy import Simulated_Galaxy
from .Simulated_Catalogue import Simulated_Catalogue
from . import useful_funcs_austind
from . import decorators



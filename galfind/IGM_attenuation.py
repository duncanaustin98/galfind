#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:27:07 2023

@author: u92876da
"""

# IGM_attenuation.py
import numpy as np

def calc_Inoue14_LS_LAF_optical_depth(wav_obs, z):
    pass

def calc_Inoue14_LS_DLA_optical_depth(wav_obs, z):
    pass

def calc_Inoue14_LC_LAF_optical_depth(wav_obs, z):
    pass

def calc_Inoue14_LC_DLA_optical_depth(wav_obs, z):
    pass

def calc_Inoue14_transmission(wav_rest, z):
    wav_obs = wav_rest * (1 + z)
    optical_depth = calc_Inoue14_LC_DLA_optical_depth(wav_obs, z) + calc_Inoue14_LC_LAF_optical_depth(wav_obs, z) + \
        calc_Inoue14_LS_DLA_optical_depth(wav_obs, z) + calc_Inoue14_LS_LAF_optical_depth(wav_obs, z)
    transmission = np.exp(-optical_depth)
    return transmission
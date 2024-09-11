#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:50:05 2023

@author: austind
"""

# calc_NIRCam_aper_corrs.py

from galfind import NIRCam

if __name__ == "__main__":
    NIRCam_instr = NIRCam().aper_corr("", "")
